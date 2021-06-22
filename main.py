import random
import sys
import argparse
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from maze import *
from env import MazeEnv
from RL import QLearning, Sarsa, Sarsa_lambda
from utils import evaluate_Q
from gui import MazeGUI


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('mode', choices=['gui', 'shell'])
parser.add_argument('--method', choices=['Q_Learning', 'Sarsa', 'Sarsa_Lambda'], default='Q_Learning')
parser.add_argument('--maze', choices=['easy', 'mid', 'hard'], default='mid', help='maze to use')
parser.add_argument('--seed', type=int, default=4096, help='random seed')
parser.add_argument('--verbose', type=str2bool, default=False, help='verbose')
parser.add_argument('--episode', type=int, default=5000, help='Number of episodes of training.')
parser.add_argument('--test_ep', type=int, default=200, help='Number of episodes of test.')
parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor. Number in range[0, 1)')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate. Number in range[0, 1)')
parser.add_argument('--e', type=float, default=1, help='Epsilon value used in the epsilon - greedy method.')
parser.add_argument('--decay', type=float, default=0.999, help='Rate at which epsilon falls. Number in range[0, 1)')
parser.add_argument('--l', type=float, default=0.5, help='weight of TD learning. Number in range [0, 1)')


def main(args):
    np.random.seed(21)
    random.seed(21)

    if args.maze == 'easy':
        MAZE = MazeEasy
    elif args.maze == 'mid':
        MAZE = MazeMid
    else:
        MAZE = MazeHard

    env = MazeEnv(MAZE.wall, MAZE.start, MAZE.end)
    Q = np.zeros((env.nS, env.nA))
    if args.mode == 'shell':
        if args.method == 'Q_Learning':
            Q_rewards = QLearning(env, Q, args.episode, args.gamma, args.lr, args.e, args.decay)
        elif args.method == 'Sarsa':
            Q_rewards = Sarsa(env, Q, args.episode, args.gamma, args.lr, args.e, args.decay)
        else:
            Q_rewards = Sarsa_lambda(env, Q, args.episode, args.gamma, args.lr, args.e, args.decay, args.l)
        rewards, steps = evaluate_Q(env, Q, args.test_ep)
        print(f"Reward: {np.mean(rewards)} \u00B1 {np.std(rewards):.4} \t Step: {np.mean(steps)}")

        plt.plot(range(args.episode), Q_rewards)
        plt.title(args.method)
        plt.ylabel('Q reward')
        plt.xlabel('step')
        plt.show()
    else:
        app = QApplication(sys.argv)
        gui = MazeGUI(env, Q, args)
        sys.exit(app.exec())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
