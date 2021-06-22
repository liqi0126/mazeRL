import time
import numpy as np


def render_single_Q(env, Q):
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.2)
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    print("Episode reward: %f" % episode_reward)


def evaluate_Q(env, Q, num_episodes=100):
    rewards = []
    steps = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            step += 1
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        steps.append(step)
    return rewards, steps

