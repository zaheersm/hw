import gym
import numpy as np

from ..utils import *

from minatar import Environment


class Task:
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):

        random_seed(seed)
        self.env = gym.make(env_id).env
        self.env.seed(seed)
        self.name = env_id
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions[0])
        if done:
            obs = self.env.reset()
        return obs, np.asarray(rew), np.asarray(done), info


class MiniAtariTask:
    def __init__(self, env_id, seed=np.random.randint(int(1e5)), sticky_action_prob=0.0):
        random_seed(seed)
        # TODO: Allow sticky_action_prob and difficulty_ramping to be set by the configuration file
        self.env = Environment(env_id, random_seed=seed, sticky_action_prob=0.0, difficulty_ramping=False)
        self.name = env_id
        self.state_dim = self.env.state_shape()
        self.action_set = self.env.minimal_action_set()
        self.action_dim = len(self.action_set)

    def reset(self):
        self.env.reset()
        return self.env.state().flatten()

    def step(self, actions):
        rew, done = self.env.act(self.action_set[actions[0]])
        obs = self.reset() if done else self.env.state()
        return obs.flatten(), np.asarray(rew), np.asarray(done), ""


class OneRoom:
    # https://arxiv.org/pdf/1810.04586.pdf
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])

        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.current_state = 0, 0
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 14, 14

    # def reset(self):
    #     self.current_state = 0, 0
    #     return np.array(self.current_state)

    def reset(self):
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]):
                self.current_state = rand_state[0], rand_state[1]
                return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny
        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return np.array([x, y]), np.asarray(1.0), np.asarray(True), ""
        else:
            return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)

    def get_eval_goal_states(self):
        return np.array([[14, 14], [0, 0], [14, 0], [0, 14]])


class TwoRooms:
    # https://arxiv.org/pdf/1810.04586.pdf
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])
        self.obstacles_map[7, 0:] = 1.0
        self.obstacles_map[7, 7], self.obstacles_map[7, 8] = 0.0, 0.0

        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.current_state = 0, 0
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 14, 14

    def reset(self):
        self.current_state = 0, 0
        return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny

        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return np.array([x, y]), np.asarray(1.0), np.asarray(True), ""
        else:
            return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)

    def get_eval_goal_states(self):
        return np.array([[6, 0], [6, 14], [8, 0], [8, 14]])


class HardMaze:
    # https://arxiv.org/pdf/1810.04586.pdf
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])

        self.obstacles_map[2, 0:6] = 1.0
        self.obstacles_map[2, 8:] = 1.0
        self.obstacles_map[3, 5] = 1.0
        self.obstacles_map[4, 5] = 1.0
        self.obstacles_map[5, 2:7] = 1.0
        self.obstacles_map[5, 9:] = 1.0
        self.obstacles_map[8, 2] = 1.0
        self.obstacles_map[8, 5] = 1.0
        self.obstacles_map[8, 8:] = 1.0
        self.obstacles_map[9, 2] = 1.0
        self.obstacles_map[9, 5] = 1.0
        self.obstacles_map[9, 8] = 1.0
        self.obstacles_map[10, 2] = 1.0
        self.obstacles_map[10, 5] = 1.0
        self.obstacles_map[10, 8] = 1.0
        self.obstacles_map[11, 2:6] = 1.0
        self.obstacles_map[11, 8:12] = 1.0
        self.obstacles_map[12, 5] = 1.0
        self.obstacles_map[13, 5] = 1.0
        self.obstacles_map[14, 5] = 1.0
        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # self.start_states = [(0, 0), (14, 0), (7, 14), (8, 6), (14, 6)]
        self.start_states = [(0, 0), (14, 0), (7, 14)]
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 9, 9

    def reset(self):
        # k = np.random.choice([0, 1, 2, 3, 4])
        k = np.random.choice([0, 1, 2])
        self.current_state = self.start_states[k]
        return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny

        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return np.array([x, y]), np.asarray(1.0), np.asarray(True), ""
        else:
            return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_prototypes(self):
        protos = [
            [3, 8],
            [10, 3],
            [4, 14],
            [7, 14],
            [9, 9],
            [14, 3],
            [7, 0],
            [0, 0],
            [0, 14],
            [3, 3],
            [14, 0],
            [14, 14]
        ]
        # [3, 3],
        # [12, 3],
        # [10, 9],
        # [3, 14],
        # [6, 14],
        # [0, 0],
        # [14, 0],
        # [6, 14],
        # [14, 3],
        return np.array(protos)

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)

    def get_eval_goal_states(self):
        return np.array([[9, 9], [0, 0], [14, 0], [7, 14]])


class OneRoomLaplace:
    # https://arxiv.org/pdf/1810.04586.pdf
    # Same as OneRoom except that start state is random in the state space and there's no goal state
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])

        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.current_state = 0, 0
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14

    def reset(self):
        rand_state = np.random.randint(low=0, high=15, size=2)
        self.current_state = rand_state[0], rand_state[1]
        return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny
        self.current_state = x, y

        return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_eval_goal_states(self):
        return np.array([[14, 14], [0, 0], [14, 0], [0, 14]])

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)


class TwoRoomsLaplace:
    # https://arxiv.org/pdf/1810.04586.pdf
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])
        self.obstacles_map[7, 0:] = 1.0
        self.obstacles_map[7, 7], self.obstacles_map[7, 8] = 0.0, 0.0

        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.current_state = 0, 0
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14

    def reset(self):
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]):
                self.current_state = rand_state[0], rand_state[1]
                return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny

        self.current_state = x, y
        return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_eval_goal_states(self):
        return np.array([[6, 0], [6, 14], [8, 0], [8, 14]])

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)


class HardMazeLaplace:
    # https://arxiv.org/pdf/1810.04586.pdf
    def __init__(self, env_id, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.name = env_id
        self.state_dim = 2
        self.action_dim = 4

        self.obstacles_map = np.zeros([15, 15])

        self.obstacles_map[2, 0:6] = 1.0
        self.obstacles_map[2, 8:] = 1.0
        self.obstacles_map[3, 5] = 1.0
        self.obstacles_map[4, 5] = 1.0
        self.obstacles_map[5, 2:7] = 1.0
        self.obstacles_map[5, 9:] = 1.0
        self.obstacles_map[8, 2] = 1.0
        self.obstacles_map[8, 5] = 1.0
        self.obstacles_map[8, 8:] = 1.0
        self.obstacles_map[9, 2] = 1.0
        self.obstacles_map[9, 5] = 1.0
        self.obstacles_map[9, 8] = 1.0
        self.obstacles_map[10, 2] = 1.0
        self.obstacles_map[10, 5] = 1.0
        self.obstacles_map[10, 8] = 1.0
        self.obstacles_map[11, 2:6] = 1.0
        self.obstacles_map[11, 8:12] = 1.0
        self.obstacles_map[12, 5] = 1.0
        self.obstacles_map[13, 5] = 1.0
        self.obstacles_map[14, 5] = 1.0
        # D, U, R, L
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.start_states = [(0, 0), (14, 0), (7, 14), (8, 6), (14, 6)]
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14

    def reset(self):
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]):
                self.current_state = rand_state[0], rand_state[1]
                return np.array(self.current_state)

    def step(self, a):
        dx, dy = self.actions[a[0]]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not int(self.obstacles_map[nx][ny]):
            x, y = nx, ny

        self.current_state = x, y
        return np.array([x, y]), np.asarray(0.0), np.asarray(False), ""

    def get_eval_goal_states(self):
        return np.array([[9, 9], [0, 0], [14, 0], [7, 14]])

    def get_eval_states(self):
        states = []
        for x in range(15):
            for y in range(15):
                if not int(self.obstacles_map[x][y]):
                    states.append([x, y])
        return np.array(states)

