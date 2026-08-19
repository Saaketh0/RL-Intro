import gymnasium as gym
from REINFORCE_JAX import Reinforce


RANGE = 100

env = gym.make_vec("CartPole-v1", max_episode_steps=5000)#, render_mode="human")
agent = Reinforce()
# visual window (“human”), get image arrays (“rgb_array”), or run without visuals (None - fastest for training)



def run():
    observations, info = env.reset()
    agent.flush_data()

    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = agent.action(observations)

        # Take the action and see what happens
        observations, reward, terminated, truncated, info = env.step(action)

        agent.insert_reward(reward)

        total_reward += reward
        episode_over = terminated or truncated

    agent.update_weights()
    return total_reward

highest = 0
for i in range(RANGE):
    total_reward = run()
    highest = max(highest,total_reward)
    if (i % (max(1,RANGE//10))) == 0:
        print(f"Episode finished! Total reward: {total_reward}")

print(f"Max: {highest}")
env.close()


"""
Benchmarking times it takes for REINFORCE to complete 10k training loops, with the maximum episode steps of 5k
Running on a Macbook Air M3
Baseline: 5210.82s user 126.10s system 99% cpu 1:29:09.98 total, Max: 5000
V2 (Batching): 2287.27s user 12.20s system 100% cpu 38:17.78 total, Max: 5000


"""
