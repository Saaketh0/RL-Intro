import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    if terminated: print("Episode was terminated")
    if truncated: print("Episode was truncated")
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
