import numpy as np
import sys


def epsilon_greed(e, Qs, nA):
    if np.random.rand() > e:
        return np.argmax(Qs)
    else:
        return np.random.randint(nA)


def QLearning(env, Q, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99):
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s = env.reset()
        while True:
            a = epsilon_greed(e, Q[s], env.nA)
            nexts, reward, done, info = env.step(a)
            Q[s][a] += lr * (reward + gamma * np.max(Q[nexts]) - Q[s][a])
            tmp_episode_reward += reward
            s = nexts
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return episode_reward


def Sarsa_lambda(env, Q, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99, l=0.5):
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        E = np.zeros((env.nS, env.nA))
        tmp_episode_reward = 0
        s = env.reset()
        a = epsilon_greed(e, Q[s], env.nA)
        while True:
            s_next, reward, done, info = env.step(a)
            a_next = epsilon_greed(e, Q[s_next], env.nA)
            delta = reward + gamma * Q[s_next][a_next] - Q[s][a]
            E[s][a] += 1
            Q += lr * delta * E
            E = gamma * l * E
            tmp_episode_reward += reward
            s, a = s_next, a_next
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return episode_reward


def Sarsa(env, Q, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99):
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s = env.reset()
        a = epsilon_greed(e, Q[s], env.nA)
        while True:
            s_next, reward, done, info = env.step(a)
            a_next = epsilon_greed(e, Q[s_next], env.nA)
            Q[s][a] += lr * (reward + gamma * Q[s_next][a_next] - Q[s][a])
            tmp_episode_reward += reward
            s, a = s_next, a_next
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return episode_reward

