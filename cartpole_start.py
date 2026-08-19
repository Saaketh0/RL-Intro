"""
This file calls no other file and is self contained.

Can be used to test that the environment works, and to see Cartpole in action.

Uses very basic logic to decide the movements.
"""

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
# visual window (“human”), get image arrays (“rgb_array”), or run without visuals (None - fastest for training)

observation, info = env.reset()

print(f"Starting observation: {observation}")
"""
Observation Space is a array with 4 values
Cart Position
Cart Velocity
Pole Angle
Pole Angular Velocity
"""

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    # Basic test for now, if cart on left side, move right and vice versa
    if observation[2] >= 0:
        action = 1
    else:
        action = 0

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    if terminated: print("Episode was terminated")
    if truncated: print("Episode was truncated")
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

"""
Important:
    make()
    step()
    .action_space()

"""
