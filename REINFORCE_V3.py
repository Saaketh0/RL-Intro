import jax.numpy as jnp
import jax
from flax import nnx


class Reinforce:
    def __init__(self,num_envs = 32,max_len = 5000):
        self.learning_rate = 0.01
        self.discount_factor = 0.99

        self.num_envs = num_envs
        self.max_len = max_len

        self.step = 0
        self.flush_data()

    def action(self, observation, policy_keys, x):
        logits = x(observation)
        actions = jax.random.categorical(policy_keys, logits)
        self.step += 1
        return actions

    def update_weights(self, trajectory, x, optimizer, baseline) -> None:
        observations, actions, rewards, done, active = trajectory

        discounted_returns = self._get_discounted_return(done, rewards)


        temp_baseline = jnp.zeros(
            (len(discounted_returns), self.num_envs),
            dtype=jnp.float32
        )
        i = jnp.zeros(self.num_envs, dtype=int)

        def calc_baseline(x, carry):
            baseline, temp_baseline, i = carry
            j = jnp.where(
                baseline[i, 1] == 0,
                discounted_returns[i,jnp.arange(self.num_envs)],
                baseline[i, 0] / (baseline[i, 1] + 1e-8) # Fixing a warning about dividing by 0
            )

            i = jnp.where(
                done[x],
                0,
                i + 1
            )
            temp_baseline = temp_baseline.at[x].set(j)
            return baseline, temp_baseline,i

        baseline, temp_baseline, i = jax.lax.fori_loop(
            0,      # start
            len(discounted_returns),      # stop
            calc_baseline,   # function
            (baseline, temp_baseline,i) # initial value
        )

        advantage = discounted_returns - temp_baseline
        baseline = self._update_baseline(discounted_returns, done, baseline)


        def loss_fn(model):

            logits = model(observations)
            # Getting the logits of every observation, and using the action we took at that period to get the log_prob of it
            log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1),
                actions[..., None],
                axis=-1
            ).squeeze(-1)

            total_error = -(log_prob * advantage).mean()
            return total_error

        loss, grads = nnx.value_and_grad(loss_fn)(x)
        optimizer.update(x, grads)
        return baseline


    def _get_discounted_return(self, resets, rewards):
        """
        There is some small error with the first value, so if training is failing come back to this,
        this error is not big enough to warrant that much attention though
        """
        results = jnp.zeros((len(rewards), self.num_envs))
        valid = jnp.zeros(self.num_envs, dtype=bool)
        g = jnp.zeros(self.num_envs, dtype=jnp.float32)


        def rewind(j, carry):
            valid, prev_g, results = carry
            i = len(rewards) - 1 - j
            valid |= resets[i]

            # jnp.where(condition, if_true, if_false)
            g = jnp.where(
                resets[i],
                rewards[i],
                rewards[i] + (prev_g * self.discount_factor)
            )

            # Makes G zero if this is the trailing unfinished env at the end
            g = jnp.where(valid, g, 0)
            results = results.at[i].set(g)

            return valid, g, results

        valid, prev_g, results = jax.lax.fori_loop(
            0,      # start
            len(rewards),      # stop
            rewind,   # function
            (valid, g, results) # initial value
        )

        return results

    def _update_baseline(self, discounted_returns,done,baseline) -> None:
        i = jnp.zeros(self.num_envs, dtype=int)

        def step(j, carry):
            baseline, i = carry

            returns = discounted_returns[j]

            # Puts the returns beside a ones column (the ones represent the count of total for the avg.)
            # [[10, 1],
            # [[20, 1],
            # [[30, 1]]
            updates = jnp.column_stack((
                returns,
                jnp.ones(self.num_envs, dtype=jnp.float32)
            ))

            # Array showing True if the current i val is not a fringe element
            valid = returns != 0
            # Multiplies all the rows that are true by the scalar 0
            # All of these rows are fringe rows
            updates = jnp.where(
                valid[:, None],
                updates,
                0,
            )

            baseline = baseline.at[i].add(updates)

            i = jnp.where(
                done[j],
                0,
                i + 1
            )
            return baseline, i

        baseline, i = jax.lax.fori_loop(
            0,
            len(discounted_returns),
            step,
            (baseline, i)
        )
        return baseline

    def flush_data(self):
        self.step = 0

class Policy_Network(nnx.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, rngs = nnx.Rngs(0)):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        # nnx.Linear needs to be initialized by random numbers, generated by nnx.Rngs(0)
        self.shared_net = nnx.Sequential(
            nnx.Linear(obs_space_dims, hidden_space1, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_space1, hidden_space2, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_space2,action_space_dims, rngs=rngs),
        )
    # Direct replacement for the forward function
    #
    def __call__(self,x):
        return self.shared_net(x)
