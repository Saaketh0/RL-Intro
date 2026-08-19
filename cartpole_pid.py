import gymnasium as gym
import random
"""
PID Controller of Cartpole Env
Best Score was 412.0, with the respective gains being:
(0.01421, 0.0056, 0.00047, 1.20185, 0.02719)
"""


env = gym.make("CartPole-v1") # , render_mode="human"
# visual window (“human”), get image arrays (“rgb_array”), or run without visuals (None - fastest for training)




def run(values):
    Kp, Kd, Ki, Pp, Pd = values

    o, info = env.reset()
    episode_over = False
    total_reward = 0

    cart_integral = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right

        cart_integral += o[0]
        pro = (Kp * o[0])
        der = (Kd * o[1])
        int = (Ki * cart_integral)
        pole_pro = (Pp * o[2])
        pole_der = (Pd * o[3])
        u = pole_pro + pole_der - (pro + der + int)
        action = 0 if u < 0 else 1

        #action = env.action_space.sample()  # Random action for now - real agents will be smarter!

        # Take the action and see what happens
        o, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    return total_reward

def average(gains,loops = 5):
    total = 0
    for _ in range(loops):
        total += run(gains)
    return total // loops

best_score = 0
best_gains = set()

for i in range(100000):
    gains = (
        round(random.uniform(0, 0.2),5),   # Kp_angle
        round(random.uniform(0, 0.5),5),    # Kd_angle
        round(random.uniform(0, 0.1),5),    # Kp_cart
        round(random.uniform(0, 2),5),    # Kd_cart
        round(random.uniform(0, 0.07),5),  # Ki_cart
    )
    score = average(gains)
    if score > best_score:
        best_score = score
        best_gains = gains

print("Best Score is ", str(best_score))
print("Best Gains is ", str(best_gains))


env.close()
