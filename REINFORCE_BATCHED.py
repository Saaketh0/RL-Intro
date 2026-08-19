"""
REINFORCE Implementation
Equation is
Weights = Prev_weights + (Return * Score Function * Learning Rate)

Return = Gt
Score Function = delta (ln pi (at | st))
Learning Rate = Alpha

Used this for help: https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/#sphx-glr-tutorials-training-agents-mujoco-reinforce-py

This is V2 of REINFORCE, implementing batching
"""


import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Reinforce:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.baseline = []

        self.learning_rate = 0.01
        self.discount_factor = 0.99

        self.x = Policy_Network(4,2)
        self.optimizer = optim.SGD(self.x.parameters(), lr=self.learning_rate)


    def action(self, observation) -> int:
        """
        Not saving the graph in memory to save space, because the NN isn't changing and we can just save the obs+action
        to reconstruct this whole thing later all at once in a batch
        """
        observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            # No .forward() because its already handled by PyTorch
            logits = self.x(observation)
        distribution = Categorical(logits=logits)
        action = distribution.sample()

        self.observations.append(observation)
        self.actions.append(action.item())
        return action.item()

    def insert_reward(self,reward) -> None:
        self.rewards.append(reward)

    def update_weights(self) -> None:
        self.optimizer.zero_grad()

        discounted_returns = self._get_discounted_return()

        baseline = []
        for i in range(len(discounted_returns)):
            if i >= len(self.baseline):
                baseline.append(discounted_returns[i])
            else:
                baseline.append(self.baseline[i][0] / self.baseline[i][1])

        advantage = discounted_returns - torch.from_numpy(np.array(baseline))

        self._update_baseline(discounted_returns)

        actions = torch.from_numpy(np.array(self.actions))
        observations = torch.from_numpy(np.array(self.observations))

        logits = self.x(observations)
        log_prob = Categorical(logits=logits).log_prob(actions)



        total_error = -(log_prob * advantage).mean()

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

        G = self.rewards[-1]
        results = [G]

        for i in range(len(self.actions)-2,-1,-1):
            G = self.rewards[i] + (results[-1] * self.discount_factor)

            results.append(G)
        results.reverse()

        return torch.from_numpy(np.array(results))

    def _update_baseline(self, discounted_returns) -> None:
        for i in range(len(self.actions)):
            if i > len(self.baseline)-1:
                self.baseline.append([discounted_returns[i],1])
            else:
                self.baseline[i][0] += discounted_returns[i]
                self.baseline[i][1] += 1

    def flush_data(self):
        self.actions = []
        self.observations = []
        self.rewards = []


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
        return self.shared_net(x)
