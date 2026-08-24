import gymnasium as gym
from TRPO import TRPO


RANGE = 10000

env = gym.make("CartPole-v1", max_episode_steps=5000)#, render_mode="human")
agent = TRPO()
# visual window (“human”), get image arrays (“rgb_array”), or run without visuals (None - fastest for training)



def run(eval = False):
    observations, info = env.reset()
    agent.flush_data()

    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        if not eval:
            action = agent.action(observations)
        else:
            action = agent.eval_action(observations)

        # Take the action and see what happens
        observations, reward, terminated, truncated, info = env.step(action)

        agent.insert_reward(reward)

        total_reward += reward
        episode_over = terminated or truncated
    if not eval:
        agent.add_weights()

    return total_reward

highest = 0
eval = []
for i in range(RANGE):
    total_reward = run()
    highest = max(highest,total_reward)
    if (i % (max(1,RANGE//10))) == 0:
        eval_reward = run(True)
        eval.append(eval_reward)
        print(f"Eval Time! Current Eval Reward: {eval_reward}")

import matplotlib.pyplot as plt

plt.plot(eval)
plt.show()

env.close()
