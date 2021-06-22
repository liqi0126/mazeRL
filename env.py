import numpy as np
from copy import deepcopy


class ACTION:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MazeEnv:
    def __init__(self, wall, start, end):
        self.wall = wall
        self.end = end
        self.start = start
        self.pos = deepcopy(start)

        self.nS = np.prod(self.wall.shape)
        self.nA = 4

        self.reset()

    def to_index(self, pos):
        return pos[0] * self.wall.shape[1] + pos[1]

    def reset(self):
        self.pos = self.start
        return self.to_index(self.pos)

    def step(self, action):
        if action == ACTION.UP:
            new_pos = np.array([self.pos[0]-1, self.pos[1]])
        elif action == ACTION.RIGHT:
            new_pos = np.array([self.pos[0], self.pos[1]+1])
        elif action == ACTION.DOWN:
            new_pos = np.array([self.pos[0]+1, self.pos[1]])
        else:
            new_pos = np.array([self.pos[0], self.pos[1]-1])

        reward = -1
        done = False
        if self.end[tuple(new_pos)]:
            reward = self.end[tuple(new_pos)]
            self.pos = new_pos
            done = True
        elif self.wall[tuple(new_pos)]:
            reward = -1
        else:
            self.pos = new_pos

        return self.to_index(self.pos), reward, done, {}
