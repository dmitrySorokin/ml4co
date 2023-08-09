import argparse
import csv
import json
import pathlib
from multiprocessing.pool import ThreadPool
import ecole as ec
import numpy as np
from time import perf_counter
from functools import partial


def evaluate_one(problem, time_limit, memory_limit, seed_instance):
    seed, instance = seed_instance
    observation_function = ObservationFunction(problem=problem)
    policy = Policy(problem=problem)

    integral_function = BoundIntegral()

    env = Environment(
        time_limit=time_limit,
        observation_function=observation_function,
        reward_function=-integral_function,  # negated integral (minimization)
        scip_params={'limits/memory': memory_limit},
    )

    # seed both the agent and the environment (deterministic behavior)
    observation_function.seed(seed)
    policy.seed(seed)
    env.seed(seed)

    # read the instance's initial primal and dual bounds from JSON file
    with open(instance.with_name(instance.stem).with_suffix('.json')) as f:
        instance_info = json.load(f)

    # set up the reward function parameters for that instance
    initial_primal_bound = instance_info["primal_bound"]
    initial_dual_bound = instance_info["dual_bound"]
    objective_offset = 0

    integral_function.set_parameters(
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
            objective_offset=objective_offset)

    # reset the environment
    observation, action_set, reward, done, info = env.reset(str(instance), objective_limit=initial_primal_bound)

    cumulated_reward = 0  # discard initial reward

    # loop over the environment
    while not done:
        action = policy(action_set, observation)
        observation, action_set, reward, done, info = env.step(action)
        cumulated_reward += reward

    return (instance.name, {
        'instance': str(instance),
        'seed': seed,
        'initial_primal_bound': initial_primal_bound,
        'initial_dual_bound': initial_dual_bound,
        'objective_offset': objective_offset,
        'cumulated_reward': cumulated_reward,
    })


def evaluate(n_workers, problem, time_limit, memory_limit, instance_files):
     jobs = [(seed, instance) for seed, instance in enumerate(instance_files)]
     with ThreadPool(n_workers) as p:
        fn = partial(evaluate_one, problem, time_limit, memory_limit)
        yield from p.imap_unordered(fn, jobs, chunksize=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'agent',
        help='Agent to evaluate.',
        choices=['strong', 'dqn', 'reinforce', 'il'],
    )
    parser.add_argument(
        'problem',
        help='Problem benchmark to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    parser.add_argument(
        '-t', '--timelimit',
        help='Episode time limit (in seconds).',
        default=argparse.SUPPRESS,
        type=float,
    )
    parser.add_argument(
        '-j',
        help='Number of jobs',
        default=16,
        type=int
    )
 
    args = parser.parse_args()

    # check the Ecole version installed
    assert ec.__version__ == "0.7.3", "Wrong Ecole version."

    print(f"Evaluating the {args.agent} task agent on the {args.problem} problem.")

    # collect the instance files
    if args.problem == 'item_placement':
        instances_path = pathlib.Path(f"instances/1_item_placement/valid/")
        results_file = pathlib.Path(f"results/{args.agent}/1_item_placement.csv")
    elif args.problem == 'load_balancing':
        instances_path = pathlib.Path(f"instances/2_load_balancing/valid/")
        results_file = pathlib.Path(f"results/{args.agent}/2_load_balancing.csv")
    elif args.problem == 'anonymous':
        instances_path = pathlib.Path(f"instances/3_anonymous/valid/")
        results_file = pathlib.Path(f"results/{args.agent}/3_anonymous.csv")

    print(f"Processing instances from {instances_path.resolve()}")
    instance_files = list(instances_path.glob('*.mps.gz'))

    if args.problem == 'anonymous': 
        # special case: evaluate the anonymous instances five times with different seeds
        instance_files = instance_files * 5

    print(f"Saving results to {results_file.resolve()}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = ['instance', 'seed', 'initial_primal_bound', 'initial_dual_bound', 'objective_offset', 'cumulated_reward']
    with open(results_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    import sys
    sys.path.insert(1, str(pathlib.Path.cwd()))

    # set up the proper agent, environment and goal for the task
    if args.agent == "strong":
        from submissions.strong.agents.dual import Policy, ObservationFunction
        from common.environments import Branching as Environment
        from common.rewards import TimeLimitDualIntegral as BoundIntegral
        time_limit = 15*60
        memory_limit = 8796093022207  # maximum
    else:
        assert False, f"agent {args.agent} not supported"
  

    # override from command-line argument if provided
    time_limit = getattr(args, "timelimit", time_limit)

    with open(results_file, mode='a') as f:
        writer = csv.DictWriter(f, fieldnames=results_fieldnames)
        print("\n      name seed initial_primal_bound initial_dual_bound objective_offset cumulated_reward")
        for j, (name, result) in enumerate(evaluate(args.j,  args.problem, time_limit, memory_limit, instance_files)):
            writer.writerow(result)
            f.flush()

            print(
                "{j:>5d} {name:<10} {seed:<5} {initial_primal_bound:<10.3f} {initial_dual_bound:<10.3f}"
                " {objective_offset:<10.3f} {cumulated_reward:<10.3f}"
                "".format(j=j, name=name, **result)
            )

