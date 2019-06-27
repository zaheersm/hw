from ..network import *
import torch
from torch import optim

import sys


class BreakoutTrueModel:
    def __init__(self, task_fn, batch_size, discount):
        self.task_copies = []
        self.discount = discount
        self.batch_size = batch_size
        for k in range(batch_size):
            self.task_copies.append(task_fn())

        self.agent = None

    def extract_state(self, task):
        ball_x = task.env.env.ball_x
        ball_y = task.env.env.ball_y
        ball_dir = task.env.env.ball_dir
        pos = task.env.env.pos
        brick_map = np.copy(task.env.env.brick_map)
        strike = task.env.env.strike
        last_x = task.env.env.last_x
        last_y = task.env.env.last_y
        terminal = task.env.env.terminal

        return ball_x, ball_y, ball_dir, pos, brick_map, strike, last_x, last_y, terminal

    def reset_state(self, task, raw_state):
        ball_x, ball_y, ball_dir, pos, brick_map, strike, last_x, last_y, terminal = raw_state
        task.env.env.ball_x = ball_x
        task.env.env.ball_y = ball_y
        task.env.env.ball_dir = ball_dir
        task.env.env.pos = pos
        task.env.env.brick_map = np.copy(brick_map)
        task.env.env.strike = strike
        task.env.env.last_x = last_x
        task.env.env.last_y = last_y
        task.env.env.terminal = terminal

    def reset_states(self, raw_states):
        for k, raw_state in enumerate(raw_states):
            self.reset_state(self.task_copies[k], raw_state)

    def step(self, actions, sim_next_states, sim_rewards, sim_dones, rollout_length=1):
        current_states = np.zeros_like(sim_next_states)
        sum_rewards = np.zeros(self.batch_size)
        rollout_lengths = np.ones(self.batch_size)

        for t, task in enumerate(self.task_copies):
            next_state, reward, sim_dones[t], _ = self.task_copies[t].step([actions[t]])
            current_states[t] = next_state
            sum_rewards[t] = reward.astype(np.float64)

        for k in range(1, rollout_length):
            actions = self.agent.policy(current_states)
            for t, task in enumerate(self.task_copies):
                if not sim_dones[t]:
                    next_state, reward, sim_dones[t], _ = self.task_copies[t].step([actions[t]])
                    current_states[t] = next_state
                    sum_rewards[t] += reward*self.discount**k
                    rollout_lengths[t] += 1

        return current_states, sum_rewards, sim_dones, rollout_lengths

    def add_experience(self, replay, states, rollout_length=1):
        dones = np.zeros(len(states)).astype(np.bool)
        task_ids = np.arange(len(states))
        for k in range(rollout_length):
            batch = []
            actions = self.agent.policy(states)
            tasks = task_ids[~dones]
            for t in tasks:
                rs = self.extract_state(self.task_copies[t])
                ns, r, d, _ = self.task_copies[t].step([actions[t]])
                batch.append([np.copy(states[t]), actions[t], r, np.copy(ns), int(d), rs])
                states[t] = ns
                dones[t] = d
            if len(batch) > 0: replay.feed_batch(batch)
            else: break

    def add_experience_v2(self, replay, states, rollout_length=1):
        # Adding the raw state of the next state to the buffer instead of the state
        dones = np.zeros(len(states)).astype(np.bool)
        task_ids = np.arange(len(states))
        for k in range(rollout_length):
            batch = []
            actions = self.agent.policy(states)
            tasks = task_ids[~dones]
            for t in tasks:
                ns, r, d, _ = self.task_copies[t].step([actions[t]])
                rs = self.extract_state(self.task_copies[t])
                batch.append([np.copy(states[t]), actions[t], r, np.copy(ns), int(d), rs])
                states[t] = ns
                dones[t] = d
            if len(batch) > 0: replay.feed_batch(batch)
            else: break
        return []

    def add_experience_v3(self, replay, states, rollout_length=1):
        # Adding the raw state of the next state to the buffer instead of the state
        dones = np.zeros(len(states)).astype(np.bool)
        task_ids = np.arange(len(states))
        batch_return = []
        for k in range(rollout_length):
            batch = []
            actions = self.agent.policy(states)
            tasks = task_ids[~dones]
            for t in tasks:
                ns, r, d, _ = self.task_copies[t].step([actions[t]])
                rs = self.extract_state(self.task_copies[t])
                batch.append([np.copy(states[t]), actions[t], r, np.copy(ns), int(d), rs])
                if k == rollout_length - 1:
                    batch_return.append([np.copy(states[t]), actions[t], r, np.copy(ns), int(d), rs])
                states[t] = ns
                dones[t] = d
            if len(batch) > 0: replay.feed_batch(batch)
            else: break
        return batch_return
    # def debug_step(self, actions, sim_next_states, sim_rewards, sim_dones, rollout_length=1):
    #
    #     current_states = np.zeros_like(sim_next_states)
    #     sum_rewards = np.zeros(self.batch_size)
    #     rollout_lengths = np.ones(self.batch_size)
    #
    #     rewards_debug = np.zeros((self.batch_size, rollout_length))
    #     for t, task in enumerate(self.task_copies):
    #         next_state, reward, sim_dones[t], _ = self.task_copies[t].step([actions[t]])
    #         current_states[t] = next_state
    #         sum_rewards[t] = reward.astype(np.float64)
    #         rewards_debug[t][0] = reward
    #
    #     id = 3
    #     fig, ar = None, None
    #     debug_plot = False
    #     if debug_plot:
    #         fig, ar = plt.subplots(4, 4)
    #         x = current_states[id].reshape(10, 10, 4)
    #         ar[0, 0].imshow(np.sum(x, axis=2))
    #
    #     for k in range(1, rollout_length):
    #         actions = self.agent.policy(current_states)
    #         for t, task in enumerate(self.task_copies):
    #             if not sim_dones[t]:
    #                 next_state, reward, sim_dones[t], _ = self.task_copies[t].step([actions[t]])
    #                 current_states[t] = next_state
    #                 sum_rewards[t] += reward*self.discount**k
    #                 rollout_lengths[t] += 1
    #                 rewards_debug[t][k] = reward
    #
    #                 if debug_plot and t == id:
    #                     x = current_states[t].reshape(10, 10, 4)
    #                     ar[int(k/4), k%4].imshow(np.sum(x, axis=2))
    #     if debug_plot:
    #         plt.show()
    #         print(rollout_lengths[id])
    #         print(rewards_debug[id])
    #         input()
    #         plt.close()
    #     if np.sum(sum_rewards) > 0.0:
    #         xz = 0
    #
    #
    #     return current_states, sum_rewards, sim_dones, rollout_lengths
    #


class SpaceInvadersTrueModel(BreakoutTrueModel):
    def __init__(self, task_fn, batch_size, discount):
        BreakoutTrueModel.__init__(self, task_fn, batch_size, discount)

    def extract_state(self, task):
        pos = task.env.env.pos
        f_bullet_map = np.copy(task.env.env.f_bullet_map)
        e_bullet_map = np.copy(task.env.env.e_bullet_map)
        alien_map = np.copy(task.env.env.alien_map)
        alien_dir = task.env.env.alien_dir
        enemy_move_interval = task.env.env.enemy_move_interval
        alien_move_timer = task.env.env.alien_move_timer
        alien_shot_timer = task.env.env.alien_shot_timer
        ramp_index = task.env.env.ramp_index
        shot_timer = task.env.env.shot_timer
        terminal = task.env.env.terminal

        return pos, f_bullet_map, e_bullet_map, alien_map, alien_dir, enemy_move_interval, alien_move_timer, alien_shot_timer, ramp_index, shot_timer, terminal

    def reset_state(self, task, raw_state):
        pos, f_bullet_map, e_bullet_map, alien_map, alien_dir, enemy_move_interval, alien_move_timer, alien_shot_timer, ramp_index, shot_timer, terminal = raw_state

        task.env.env.pos = pos
        task.env.env.f_bullet_map = np.copy(f_bullet_map)
        task.env.env.e_bullet_map = np.copy(e_bullet_map)
        task.env.env.alien_map = np.copy(alien_map)
        task.env.env.alien_dir = alien_dir
        task.env.env.enemy_move_interval = enemy_move_interval
        task.env.env.alien_move_timer = alien_move_timer
        task.env.env.alien_shot_timer = alien_shot_timer
        task.env.env.ramp_index = ramp_index
        task.env.env.shot_timer = shot_timer
        task.env.env.terminal = terminal