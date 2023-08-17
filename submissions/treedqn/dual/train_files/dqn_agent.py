
import torch
from torch import nn
import numpy as np
import copy
import sys

sys.path.insert(0,'.')
from model import GNNPolicy


class DQNAgent:
    def __init__(self, device, epsilon):
        super().__init__()
        self.net = GNNPolicy(device=device).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        self.target_net_freq = 1000
        self.grad_norm = 50
        self.gamma = 1.0
        self.epsilon = epsilon

    def _predict(self, qnet, obs):
        value, adv = qnet(obs)
        logit = value.mean() + (adv - adv.mean())
        return torch.exp(logit)

    def _act(self, qnet, obs, action_set):
        with torch.no_grad():
            preds = self._predict(qnet, obs)[action_set.astype('int32')]

        action_idx = torch.argmax(preds)
        action = action_set[action_idx.item()]
        return action

    def act(self, obs, action_set, deterministic):
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.choice(action_set)

        return self._act(self.net, obs, action_set)

    def loss(self, obs, nextobs, next_actset, reward, action, done):
        value, adv = self.net(obs)
        logq_pred = (value.mean() + (adv - adv.mean()))[action]

        next_actions = [self._act(self.net, nobs, nactset) for nobs, nactset in zip(nextobs, next_actset)]
        q_next = torch.zeros_like(logq_pred) + sum([
            self._predict(self.target_net, nobs)[naction].detach() for nobs, naction in zip(nextobs, next_actions)
        ])

        q_fact = reward + q_next * self.gamma * (1 - done)
        q_fact = torch.maximum(q_fact, torch.tensor(1e-5))
        # print(reward, q_next, done, self.gamma, q_fact, logq_pred)
        assert q_fact == q_fact, f'{reward}, {q_next}, {done}, {self.gamma}, {q_fact}, {logq_pred}'
        return (logq_pred - torch.log(q_fact)) ** 2

    def update(self, step, batch):
        self.optimizer.zero_grad()

        loss = 0
        
        for trans in zip(
            batch['obs'], batch['next_obs'], batch['next_actset'],
            batch['rew'], batch['act'], batch['done']):
            loss += self.loss(*trans)
        
        loss /= len(batch['obs'])
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm)
        self.optimizer.step()

        if step % self.target_net_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss.detach().cpu().item()

    
    def save(self, name):
        torch.save(self.net.state_dict(), name)

    def load(self, path):
        self.net.load_state_dict(
            torch.load(path, map_location=self.net.device)
        )

    def load_il(self, path):
        self.net.load_il(
            torch.load(path, map_location=self.net.device)
        )
        self.target_net.load_il(
            torch.load(path, map_location=self.target_net.device)
        )

    def train(self):
        self.net.train()
        self.target_net.train()

    def eval(self):
        self.net.eval()
