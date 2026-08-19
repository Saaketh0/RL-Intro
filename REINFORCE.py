"""
REINFORCE Implementation
Equation is
Weights = Prev_weights + (Return * Score Function * Learning Rate)

Return = Gt
Score Function = delta (ln pi (at | st))
Learning Rate = Alpha

Used this for help: https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/#sphx-glr-tutorials-training-agents-mujoco-reinforce-py
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random


class Reinforce:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.reward = []
        self.baseline = []

        self.learning_rate = 0.01
        self.discount_factor = 0.99

        self.x = Policy_Network(4,2)
        self.optimizer = optim.SGD(self.x.parameters(), lr=self.learning_rate)


    def action(self, observation) -> int:
        sample = self.x.forward(observation)

        distribution = Categorical(logits=sample)
        action = distribution.sample()

        self.actions.append(action.item())
        """ The log prob is basically ln(π(at∣st)) """
        self.log_probs.append(distribution.log_prob(action))
        #print(distribution)
        return action.item()

    def insert_reward(self,reward) -> None:
        self.reward.append(reward)

    def update_weights(self) -> None:
        self.optimizer.zero_grad()
        total_error = 0
        tensor_list = []
        discounted_returns = self._get_discounted_return()

        for i in range(len(self.actions)):
            if i > len(self.baseline)-1:
                self.baseline.append([discounted_returns[i],1])
                baseline = discounted_returns[i]
            else:
                baseline = self.baseline[i][0] // self.baseline[i][1]
                self.baseline[i][0] += discounted_returns[i]
                self.baseline[i][1] += 1

            tensor_list.append(
                -1 * (discounted_returns[i] - baseline) * self.log_probs[i]
            )

        total_error = torch.stack(tensor_list).mean()

        total_error.backward()
        self.optimizer.step()

    def _get_discounted_return(self) -> list:
        """
        G(t) = Rt + γRt+1 + γ^2Rt+2 + ... + γ^nRt+n
        Rt = reward at t
        γ = discount factor
        """
        """
        This backwards method allows for you to calculate everything all at once, using the principle of the
        i'th G value just being
        Gi = Ri + γ(Ri+1)
        So we calculate it backward
        """

        G = self.reward[-1]
        results = [G]

        for i in range(len(self.actions)-2,-1,-1):
            G = self.reward[i] + (results[-1] * self.discount_factor)

            results.append(G)
        results.reverse()

        return results

    def flush_data(self):
        self.actions = []
        self.log_probs = []
        self.reward = []


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2,action_space_dims),
        )
    def forward(self,x):
        return self.shared_net(torch.tensor(x))
