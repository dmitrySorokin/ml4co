import numpy as np
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp
import time
import torch
import ecole
import json
from pathlib import Path
from ecole.scip import Model
from argparse import ArgumentParser
import glob
import typing
import sys

sys.path.append('../../..')
from common.environments import Branching as Environment, BranchingDynamics
from common.rewards import TimeLimitDualIntegral as BoundIntegral

from utilities import ChildObservation
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent



class DFSBranchingDynamics(BranchingDynamics):
    """
    Custom branching environment that changes the node strategy to DFS when training.
    """
    def reset_dynamics(self, model, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if training:
            # Set the dfs node selector as the least important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 666666)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 666666)
        else:
            # Set the dfs node selector as the most important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 0)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 0)

        return super().reset_dynamics(model, *args, **kwargs)


class EcoleBranching(Environment):
    __Dynamics__ = DFSBranchingDynamics
    
    def __init__(self, time_limit, training):
        self.training = training
    
        observation_function = ecole.observation.NodeBipartite()

        information_function = {
            'children': ChildObservation(),
            'num_nodes': ecole.reward.NNodes().cumsum(),
            'lp_iterations': ecole.reward.LpIterations().cumsum(),
            'solving_time': ecole.reward.SolvingTime().cumsum(),
        }

        self.integral_function = BoundIntegral()

        if training:
            super().__init__(
                time_limit=time_limit, # time limit for solving each instance
                observation_function=observation_function,
                information_function=information_function,
                reward_function=-self.integral_function,
            )
        else:
            super().__init__(
                time_limit=time_limit, # time limit for solving each instance
                observation_function=observation_function,
                reward_function=-self.integral_function,
           )


    def reset(self, instance: Path):
        with open(str(instance)[:-6] + 'json') as f:
            instance_info = json.load(f)

        # set up the reward function parameters for that instance
        initial_primal_bound = instance_info["primal_bound"]
        initial_dual_bound = instance_info["dual_bound"]
        objective_offset = 0

        self.integral_function.set_parameters(
                initial_primal_bound=initial_primal_bound,
                initial_dual_bound=initial_dual_bound,
                objective_offset=objective_offset)
        
        task = Model.from_file(str(instance))
        return super().reset(task, training=self.training, objective_limit=initial_primal_bound)


class EvalProcess(mp.Process):
    def __init__(self, device, valid_tasks, in_queue, out_queue):
        super().__init__()
        self.device = device
        self.tasks = valid_tasks
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        env = EcoleBranching(time_limit=15 * 60, training=False)
        agent = DQNAgent(device=self.device, epsilon=0)
        stop = False
        while not stop:
            if self.in_queue.empty():
                time.sleep(60)
                continue
            (checkpoint, episode, stop) = self.in_queue.get()

            agent.load(checkpoint)
            n_nodes, returns = [], []
            for instance in self.tasks:
                cum_reward = 0
                env.seed(0)
                obs, act_set, reward, done, _ = env.reset(instance)
                cum_reward += reward
                while not done:
                    act = agent.act(obs, act_set, deterministic=True)
                    obs, act_set, reward, done, _ = env.step(act)
                    cum_reward += reward
                n_nodes.append(env.model.as_pyscipopt().getNNodes())
                returns.append(cum_reward)
            geomean = np.exp(np.mean(np.log(n_nodes)))
            self.out_queue.put(('eval/nnodes', geomean, episode))
            self.out_queue.put(('eval/return', np.mean(returns), episode))


def rollout(env, agent, replay_buffer, instances, seed, rng, max_tree_size=1000):
    instance = Path(rng.choice(instances))

    env.seed(seed)
    obs, act_set, reward, done, info = env.reset(instance)
    
    cum_reward = reward
    traj_obs, traj_rew, traj_act, traj_actset, traj_done = [], [], [], [], []
    while not done:
        action = agent.act(obs, act_set, deterministic=False)
        
        traj_obs.append(obs)
        traj_rew.append(reward)
        traj_act.append(action)
        traj_actset.append(act_set)
        
        obs, act_set, reward, done, info = env.step(action)
        traj_done.append(done)
        cum_reward += reward

    traj_nextobs, traj_nextactset = [], []
    for children in info['children']:
        traj_nextobs.append([traj_obs[c] for c in children])
        traj_nextactset.append([traj_actset[c] for c in children])

    assert len(traj_obs) == len(traj_nextobs)
    tree_size = len(traj_obs)
    ids = np.random.choice(range(tree_size), min(tree_size, max_tree_size), replace=False)
    
    # ids = list(range(min(tree_size, max_tree_size)))
    traj_obs = np.asarray(traj_obs)[ids]
    traj_rew = np.asarray(traj_rew)[ids]
    traj_act = np.asarray(traj_act)[ids]
    
    traj_nextobs = np.array(traj_nextobs, dtype=list)[ids]
    traj_nextactset = np.array(traj_nextactset, dtype=list)[ids]
    traj_done = np.asarray(traj_done)[ids]


    for transition in zip(traj_obs, traj_rew, traj_act, traj_nextobs, traj_nextactset, traj_done):
        replay_buffer.add_transition(*transition)

    info['return'] = cum_reward
    return len(traj_obs), info


def main(cfg: typing.Dict):
    parser = ArgumentParser()
    parser.add_argument('problem')

    args = parser.parse_args()

    # get instances
    if args.problem == 'item_placement':
        instances_train = glob.glob('../../../instances/1_item_placement/train/*.mps.gz')
        instances_valid = glob.glob('../../../instances/1_item_placement/valid/*.mps.gz')
        out_dir = 'train_files/samples/1_item_placement'

    elif args.problem == 'load_balancing':
        instances_train = glob.glob('../../../instances/2_load_balancing/train/*.mps.gz')
        instances_valid = glob.glob('../../../instances/2_load_balancing/valid/*.mps.gz')
        out_dir = 'train_files/samples/2_load_balancing'

    elif args.problem == 'anonymous':
        instances_train = glob.glob('../../../instances/3_anonymous/train/*.mps.gz')
        instances_valid = glob.glob('../../../instances/3_anonymous/valid/*.mps.gz')
        out_dir = 'train_files/samples/3_anonymous'

    else:
        raise NotImplementedError
    
    mp.set_start_method('spawn')
    writer = SummaryWriter(out_dir)


    # randomization setup
    rng = np.random.RandomState(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    

    env = EcoleBranching(time_limit=5 * 60, training=True)

    agent = DQNAgent(device=cfg['device'], epsilon=1)
    agent.train()

    replay_buffer = ReplayBuffer(
        max_size=cfg['buffer_max_size'],
        start_size=cfg['buffer_start_size']
    )

    pbar = tqdm(total=replay_buffer.start_size, desc='init')
    while not replay_buffer.is_ready():
        num_obs, _ = rollout(env, agent, replay_buffer, instances_train, cfg['seed'], rng)
        pbar.update(num_obs)
    pbar.close()

    pbar = tqdm(total=cfg['num_episodes'], desc='train')
    update_id = 0
    episode_id = 0
    epsilon_min = 0.01
    decay_steps = cfg['decay_steps']

    in_queue, out_queue = mp.Queue(), mp.Queue()
    evaler = EvalProcess(cfg['device'], instances_valid, in_queue, out_queue)
    evaler.start()

    while episode_id < pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer, instances_train, cfg['seed'], rng)
        writer.add_scalar('episode/num_nodes', info['num_nodes'], episode_id)
        writer.add_scalar('episode/lp_iterations', info['lp_iterations'], episode_id)
        writer.add_scalar('episode/solving_time', info['solving_time'], episode_id)
        writer.add_scalar('episode/return', info['return'], episode_id)
        writer.add_scalar('train/epsilon', agent.epsilon, episode_id)

        print(f" ep: {episode_id}, nnodes: {info['num_nodes']}, ret: {info['return']}")

        episode_loss = []
        for i in range(num_obs):
            episode_loss.append(agent.update(update_id, replay_buffer.sample()))
            update_id += 1
        
        writer.add_scalar('train/loss', np.mean(episode_loss), episode_id)

        chkpt = out_dir + f'/checkpoint_{episode_id}.pkl'
        agent.save(chkpt)

        if episode_id % cfg['eval_freq'] == 0 or episode_id == cfg['num_episodes']:
            chkpt = out_dir + f'/checkpoint_{episode_id}.pkl'
            agent.save(chkpt)
            in_queue.put((chkpt, episode_id, episode_id == cfg['num_episodes']))

        episode_id += 1
        epsilon = 1. - (1. - epsilon_min) / decay_steps * update_id
        agent.epsilon = max(epsilon_min, epsilon)

        while not out_queue.empty():
            writer.add_scalar(*out_queue.get_nowait())


        pbar.update(1)
    evaler.join()
    pbar.close()

    while not out_queue.empty():
        writer.add_scalar(*out_queue.get_nowait())


if __name__ == '__main__':
    cfg = {
        'device': 'cuda:0',
        'seed': 0,
        'buffer_max_size': int(1e4),
        'buffer_start_size': int(1e3),
        'num_episodes': 100,
        'eval_freq': 10,
        'decay_steps': int(1e5),
    }

    main(cfg)
