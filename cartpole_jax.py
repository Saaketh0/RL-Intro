from REINFORCE_JAX import Reinforce, Policy_Network
import jax
import jax.numpy as jnp
import gymnax
from flax import nnx
import optax


RANGE = 100
MAX_LEN = 5000
BATCH_SIZE = 8192

key = jax.random.key(0)
key, key_reset, key_policy, key_step = jax.random.split(key, 4)
reset_keys = jax.random.split(key_reset, BATCH_SIZE)


env, env_params = gymnax.make("CartPole-v1")
env_params = env_params.replace(max_steps_in_episode=MAX_LEN)

x = Policy_Network(4,2)
optimizer = nnx.Optimizer(
    x,
    optax.sgd(0.001),
    wrt=nnx.Param
)
baseline = jnp.zeros((MAX_LEN, 2), dtype=jnp.float32)

agent = Reinforce(BATCH_SIZE, MAX_LEN)
# visual window (“human”), get image arrays (“rgb_array”), or run without visuals (None - fastest for training)
#
@nnx.jit
def run(key, x, optimizer, baseline):
    key, key_reset, key_policy, key_step = jax.random.split(key, 4)
    reset_keys = jax.random.split(key_reset, BATCH_SIZE)
    observations, state = jax.vmap(
        env.reset,
        in_axes=(0, None)
    )(reset_keys, env_params)
    agent.flush_data()

    episode_over = jnp.zeros(BATCH_SIZE, dtype=bool)
    total_reward = jnp.zeros(BATCH_SIZE, dtype=jnp.float32)

    def step(carry,y):
        key, observations, state, episode_over, total_reward = carry
        key, key_policy, key_step = jax.random.split(key, 3)

        old_observations = observations

        action_space = env.action_space(env_params)
        policy_keys = jax.random.split(key_policy, BATCH_SIZE)
        action = jax.vmap(
            agent.action,
            in_axes=(0,0,None)
        )(observations, policy_keys, x)

        # Take the action and see what happens
        step_keys = jax.random.split(key_step, BATCH_SIZE)
        observations, state, reward, done, _ = jax.vmap(
            env.step,
            in_axes=(0, 0, 0, None)
        )(step_keys, state, action, env_params)

        active = ~episode_over
        raw_reward = reward
        total_reward += jnp.where(active, raw_reward, 0.0)
        episode_over = episode_over | done
        trajectory_step = (old_observations, action, raw_reward, done, active)
        return (key, observations, state, episode_over, total_reward), trajectory_step

    initial_carry = (
        key,
        observations,
        state,
        episode_over,
        total_reward,
    )
    # the xs is the input, which considering we are not changing anything we set it to be empty, represented at y inside
    carry, trajectory = jax.lax.scan(step, initial_carry, xs=None, length=MAX_LEN)
    key, observations, state, episode_over, total_reward = carry

    baseline = agent.update_weights(trajectory, x, optimizer, baseline)
    return total_reward, key, trajectory, baseline

highest = 0
for i in range(RANGE):
    total_reward, key, trajectory, baseline = run(key, x, optimizer, baseline)
    highest = max(highest, jnp.max(total_reward))
    if (i % (max(1,RANGE//10))) == 0:
        observations, actions, rewards, done, active = trajectory
        print(f"Total Reward: {total_reward}")
    #if MAX_STEP_COUNt < 0:
    #    print(f"Max step count reached {MAX_STEP_COUNt} times! Exiting Prematurely.")
print(f"Max: {highest}")
#nv.close()

"""
Baseline: 5.04s user 0.35s system 103% cpu 5.212 total
Basic MPS Integration: 6.72s user 1.63s system 60% cpu 13.768 total

"""
