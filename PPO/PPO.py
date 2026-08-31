"""
V1 of my PPO Implementation from scratch

Ok, so we are taking parts of the TRPO equation that did not change.
For example, the probability ratio between the old and new policy

"PPO does not clip every big policy change. It only clips a big policy change when that change is helping the objective too much."

When a ratio between new/old, the objective stops increasing for that specific action. It still prefers it but it doesn't just increase.

"While I am reusing this batch, don't let the current policy exploit this old data by moving too far away from the policy that originally generated it."


"""


import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class PPO:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.baseline = []
        self.batch = []

        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 0.2 # This is the clip inside PPO

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
        action = distribution.sample().item()

        self.observations.append(observation)
        self.actions.append(action)
        return action

    def eval_action(self, observation) -> int:
        # Only for evals, picks the most likely outcome, no sampling
        observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            logits = self.x(observation)
        # the .item() converts a 0D tensor into an int
        return torch.argmax(logits).item()

    def insert_reward(self, reward) -> None:
        self.rewards.append(reward)

    def update_weights(self) -> None:
        self.optimizer.zero_grad()

        # All the code from 95-119 is unpacking the accumulated episode data, and calculating advantage
        all_observations = []
        all_actions = []
        all_advantages = []

        for observations, actions, rewards in self.batch:
            discounted_returns = self._get_discounted_return(rewards)

            baseline = []
            for i in range(len(discounted_returns)):
                if i >= len(self.baseline):
                    baseline.append(discounted_returns[i])
                else:
                    baseline.append(self.baseline[i][0] / self.baseline[i][1])
            advantage = discounted_returns - torch.from_numpy(np.array(baseline))
            self._update_baseline(discounted_returns)

            all_observations.extend(observations)
            all_actions.extend(actions)
            all_advantages.extend(advantage)

        observations = torch.tensor(np.array(all_observations))
        actions = torch.tensor(np.array(all_actions))
        advantage = torch.tensor(np.array(all_advantages))

        # This is freezing the old NN weights, to be used as a constant. (Used in denom in future equation)
        old_policy = copy.deepcopy(self.x)
        for param in old_policy.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            old_logits = old_policy(observations)
            old_dist = Categorical(logits=old_logits)

            old_action_probs = old_dist.probs.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

        # You are calculating the new action probabilities, we already calculated these in the action() step, but I am recomputing here to make it obvious
        # The difference between these and the old policy is that the old policy distribution are genuine numbers, and not changeable, these are linked to the NN, so the gradients will get computed.
        # # TLDR: old_policy have constant values, these are variables that can change
        logits = self.x(observations)
        new = Categorical(logits=logits)
        new_action_probs = new.probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # This is the part in the equation: theta_new / theta_old
        new_over_old = new_action_probs / old_action_probs

        """
        This is the core additions of PPO down below, the clipping function
        """

        unclipped = advantage * new_over_old
        # torch.clamp does target, min, max
        clipped = advantage * torch.clamp(new_over_old, 1-self.epsilon, 1+self.epsilon)
        l = torch.min(unclipped, clipped)

        total_error = -l.mean()
        total_error.backward()

        self.optimizer.step()

    def add_weights(self) -> None:
        self.batch.append((
            self.observations.copy(),
            self.actions.copy(),
            self.rewards.copy()
        ))
        self.observations = []
        self.actions = []
        self.rewards = []

        if len(self.batch) == 1:
            self.update_weights()
            self.batch = []

    def _get_discounted_return(self, rewards):
        g = rewards[-1]
        results = [g]

        for i in range(len(rewards)-2,-1,-1):
            g = rewards[i] + (results[-1] * self.discount_factor)

            results.append(g)
        results.reverse()

        return torch.from_numpy(np.array(results))

    def _update_baseline(self, discounted_returns) -> None:
        for i in range(len(discounted_returns)):
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
