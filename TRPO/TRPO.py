"""
TRPO Implementation

The Equation for Hessian KL Divergence = sqrt( 2δ / x^T * H * x)

Note: .flatten() stacks the 2D tensor into a 1D tensor
"""


import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class TRPO:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.baseline = []
        self.batch = []

        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.KL_Divergence = 0.001

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
        # With vector p, you are finding the Hessian
        def hessian_vector_product(p):
            new_logits = self.x(observations)
            log_p = torch.log_softmax(old_logits, dim=1)
            log_g = torch.log_softmax(new_logits, dim=1)
            old_probs = old_dist.probs
            KL = torch.sum(
                old_probs * (log_p - log_g),
                dim=1
            ).mean()
            # The first derivative
            kl_grads = torch.autograd.grad(
                KL,
                self.x.parameters(),
                create_graph=True
            )
            # Flattening it out
            flat_kl_grad = torch.cat([
                grad.flatten()
                for grad in kl_grads
            ])
            # Calculating the second order derivatives for just the g values
            grad_dot_p = torch.dot(flat_kl_grad, p)

            hvp = torch.autograd.grad(
                grad_dot_p,
                self.x.parameters()
            )

            return torch.cat([
                grad.flatten()
                for grad in hvp
            ])

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

        # This is the loss equation, although the new/old feel like they are the same, old are frozen constants while new are variables linked to PyTorches execution graph
        total_error = -(new_over_old * advantage).mean()


        total_error.backward()

        # g is the gradient vector taken from the .backward() step
        g = torch.cat([
            param.grad.flatten()
            for param in self.x.parameters()
        ])

        # This below is the KL Divergence, at the top of this function is another function that calculates the Hessian * x from vector x.
        # We are not calculating the entire Hessian because that is compute intensive, so we are performing an approximation to find x, the optimal movement for this

        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()

        for _ in range(10):
            Hp = hessian_vector_product(p)

            # Given the remaining error and the curvature in this direction, how big should this correction be?
            alpha = (torch.dot(r,r) / (torch.dot(p,Hp) + 1e-8))

            # Move my estimate of H−1g some amount along p.
            x += alpha * p

            # How wrong is my current x after that update?
            new_r = r - alpha * Hp

            p = new_r + torch.dot(new_r, new_r) / (torch.dot(r, r) + 1e-8) * p
            r = new_r

        Hx = hessian_vector_product(x)
        # x = H^-1 * g

        KL_scalar = torch.sqrt(
            (2 * self.KL_Divergence) /
            (torch.dot(x, Hx) + 1e-8) # xT * H * x
        )

        # Now instead of doing optimizer.step(), we are manually changing the weights by the gradient change
        # x is one flattened vector with all the changes, but self.x is not a single vector, so we are updating the param weights by their size sequentially
        # Remember, x is our gradient, it used to be g, but x is more optimal (x could also just == g too)
        index = 0
        with torch.no_grad():
            for param in self.x.parameters():
                num_params = param.numel()

                param_direction = x[index:index + num_params].view_as(param)

                param -= KL_scalar * param_direction

                index += num_params

    def add_weights(self) -> None:
        self.batch.append((
            self.observations.copy(),
            self.actions.copy(),
            self.rewards.copy()
        ))
        self.observations = []
        self.actions = []
        self.rewards = []

        if len(self.batch) == 10:
            self.update_weights()
            self.batch = []

    def _get_discounted_return(self, rewards):
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
