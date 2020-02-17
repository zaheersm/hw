from ..network import *
from ..component import *
from .base_agent import *

import matplotlib.pyplot as plt
import seaborn as sns


class RNDAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.q_replay = config.q_replay_fn()
        self.q_net = config.q_net_fn()
        self.q_target_net = config.q_net_fn()
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.q_opt = config.q_opt_fn(self.q_net.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.q_replay.batch_size, config.device)

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.num_episodes = 0

        self.eval_number = 0

        self.state_visitation = np.zeros([15, 15])
        self.rnd_target = config.rnd_fn()
        self.rnd_predictior = config.rnd_fn()
        self.rnd_opt = config.rnd_opt_fn(self.rnd_predictior.parameters())
        self.action_count = np.zeros([self.config.action_dim])

    def q_policy(self, phi_s):
        q_values = self.q_net(phi_s)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < self.config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        return action

    def step(self):
        config = self.config
        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        x, y = self.state
        self.state_visitation[int(x)][int(y)] += 1

        phi_s = config.state_normalizer(self.state)
        phi_s = config.rnd_state_normalizer(phi_s)
        action = self.q_policy(phi_s)

        next_state, reward, done, info = self.task.step([action])
        phi_ns = config.state_normalizer(next_state)
        phi_ns = config.rnd_state_normalizer(phi_ns)

        reward = self.get_reward(phi_ns)
        reward = config.reward_normalizer(reward)

        entry = [phi_s, action, reward, phi_ns, int(done), info]
        self.state = next_state
        state, action, reward, next_state, done, _ = entry
        self.q_replay.feed_batch([[state, action, reward, next_state, done]])

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1

        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True
            self.num_episodes += 1

        experiences = self.q_replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        q_next = self.q_target_net(next_states) if config.use_target_network else self.q_target_net(next_states)
        q_next = q_next.detach().max(1)[0]
        terminals = tensor(terminals, self.config.device)
        rewards = tensor(rewards, self.config.device)
        q_next = self.config.discount * q_next * (1 - terminals).float()
        q_next.add_(rewards.float())
        actions = tensor(actions, self.config.device).long()
        q = self.q_net(states)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

        if config.use_target_network and self.total_steps % self.config.target_network_update_freq == 0:
            self.q_target_net.load_state_dict(self.q_net.state_dict())

        self.train_rnd_network()

    def train_rnd_network(self):
        experiences = self.q_replay.sample()
        states, _ , _, _, _ = experiences
        with torch.no_grad():
            target = self.rnd_target(states)
        predictions = self.rnd_predictior(states)

        mse = (target-predictions).pow(2)
        loss = mse.mul(0.5).mean()
        self.rnd_opt.zero_grad()
        loss.backward()
        self.rnd_opt.step()

    def get_reward(self, phi_ns):
        return ((self.rnd_predictior(phi_ns) - self.rnd_target(phi_ns))**2).item()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        phi = self.config.state_normalizer(state)
        q = self.q_net(phi)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def load(self, dir):
        path = os.path.join(dir, "vec")
        self.vectors.load_state_dict(torch.load(path))

    def save(self, filename):
        pass

    def eval_episodes(self):
        super(RNDAgent, self).eval_episodes()
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        # states = self.task.get_eval_states()
        # goals = self.task.get_eval_goal_states()
        #
        # states_ = tensor(self.config.state_normalizer(states), self.config.device)
        # goals_ = tensor(self.config.state_normalizer(goals), self.config.device)
        #
        # with torch.no_grad():
        #     out = self.vectors(torch.cat([states_, goals_]))
        # f_s = out[:len(states_)]
        # f_g = out[len(states_):]
        #
        # f_s = f_s.unsqueeze(2)
        # f_g = f_g.unsqueeze(2)
        #
        # fig, ax = plt.subplots(nrows=len(goals), ncols=1, figsize=(6, 6 * 4))
        # for g_k in range(len(goals)):
        #     g = f_g[g_k]
        #     l2_vec = (f_s - g)**2
        #     l2_vec = torch.sum(l2_vec.squeeze(2), 1)
        #     distance = np.zeros((15, 15))
        #     for k, s in enumerate(states):
        #         x, y = s
        #         distance[x][y] = l2_vec[k].item()
        #     sns.heatmap(distance, ax=ax[g_k])
        #     ax[g_k].set_title('Goal: {}, {}'.format(goals[g_k][0], goals[g_k][1]))
        # plt.savefig(os.path.join(heatmap_dir, 'heatmap_{}.png'.format(self.eval_number)))
        # plt.close()
        #
        # fig, ax = plt.subplots(nrows=self.config.d, ncols=1, figsize=(6, 6 * self.config.d))
        # for d in range(self.config.d):
        #     values = np.zeros((15, 15))
        #     for k, s in enumerate(states):
        #         v = f_s[k][d].item()
        #         x, y = s
        #         values[x][y] = v
        #     sns.heatmap(values, ax=ax[d])
        #     ax[g_k].set_title('d: {}'.format(d))
        # plt.savefig(os.path.join(heatmap_dir, 'components_{}.png'.format(self.eval_number)))
        # plt.close()
        #

        if self.eval_number > 0:
            plt.plot()
            sns.heatmap(self.state_visitation/np.sum(self.state_visitation), vmin=0.0, vmax=0.01)
            plt.savefig(os.path.join(heatmap_dir, 'state_visitation_{}.png'.format(self.eval_number)))
            plt.close()
        self.state_visitation = np.zeros([15, 15])
        self.eval_number +=1
