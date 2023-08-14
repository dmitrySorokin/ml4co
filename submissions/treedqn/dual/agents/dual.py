import torch
import ecole as ec
import numpy as np

from model import GNNPolicy


class ObservationFunction(ec.observation.NodeBipartite):

    def __init__(self, problem):
        super().__init__()

    def seed(self, seed):
        pass


class Policy():

    def __init__(self, problem):
        self.rng = np.random.RandomState()

        # get parameters
        params_path = f'agents/trained_models/{problem}/best_params.pkl'

        # set up policy
        self.device = f"cuda:0"
        self.net = GNNPolicy().to(self.device)
        self.net.load_state_dict(torch.load(params_path))

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        value, adv = self.net(observation)
        qvalue = -torch.exp(value.mean() + (adv - adv.mean()))

        with torch.no_grad():
            preds = qvalue[action_set.astype('int32')]

        action_idx = torch.argmax(preds)
        action = action_set[action_idx.item()]
        return action
