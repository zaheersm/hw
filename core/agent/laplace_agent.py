from ..network import *
from ..component import *
from .base_agent import *

import matplotlib.pyplot as plt
import seaborn as sns


class LaplaceRepresentationAgent(Agent):
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

        self.l_replay = config.l_replay_fn()
        self.vectors = config.vector_fn()
        self.l_opt = config.l_opt_fn(self.vectors.parameters())
        self.trajectory = []

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1),
                                  p=self.tau_probs_norm, size=config.l_batch_size)
        self.num_episodes = 0

        self.total_loss = []
        self.attractive_loss = []
        self.repulsive_loss = []

        self.eval_number = 0

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        s = config.state_normalizer(self.state)
        self.trajectory.append(s)

        phi_s = self.get_fs(tensor(s, config.device).unsqueeze(0))
        q_values = self.q_net(phi_s)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])

        ns = config.state_normalizer(next_state)
        phi_ns = self.get_fs(tensor(ns, config.device).unsqueeze(0))
        entry = [phi_s, action, reward, phi_ns, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True
            self.trajectory.append(config.state_normalizer(self.state))
            self.l_replay.feed_batch([np.array(self.trajectory)])
            self.trajectory = []
            self.num_episodes += 1
            self.train_representation()
        self.q_replay.feed_batch([[state, action, reward, next_state, done]])

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

    def get_fs(self, s):
        fs = self.vectors(s).view(self.config.d).cpu().detach().numpy()
        return fs

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        phi = self.get_fs(state)
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
        super(LaplaceRepresentationAgent, self).eval_episodes()
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        with torch.no_grad():
            out = self.vectors(torch.cat([states_, goals_]))
        f_s = out[:len(states_)]
        f_g = out[len(states_):]

        f_s = f_s.unsqueeze(2)
        f_g = f_g.unsqueeze(2)

        fig, ax = plt.subplots(nrows=len(goals), ncols=1, figsize=(6, 6 * 4))
        for g_k in range(len(goals)):
            g = f_g[g_k]
            l2_vec = (f_s - g)**2
            l2_vec = torch.sum(l2_vec.squeeze(2), 1)
            distance = np.zeros((15, 15))
            for k, s in enumerate(states):
                x, y = s
                distance[x][y] = l2_vec[k].item()
            sns.heatmap(distance, ax=ax[g_k])
            ax[g_k].set_title('Goal: {}, {}'.format(goals[g_k][0], goals[g_k][1]))
        self.eval_number +=1
        plt.savefig(os.path.join(heatmap_dir, 'heatmap_{}.png'.format(self.eval_number)))
        plt.close()

    def train_representation(self):
        bs, d, to, dev, = self.config.l_batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.l_replay.sample_array()
        samples = self.sample_tau()
        u, v = [], []
        for k in range(self.config.l_batch_size):
            ep = batch[k]
            # if len(ep) < 2: continue
            u_idx = np.random.randint(0, len(ep) - 1)
            v_idx = min(u_idx + samples[k], len(ep) - 1)
            u_ = ep[u_idx]
            v_ = ep[v_idx]
            u.append(u_)
            v.append(v_)

        u = tensor(u, dev)
        v = tensor(v, dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        batch_u = self.l_replay.sample_array()
        batch_v = self.l_replay.sample_array()
        u, v = [], []
        for k in range(self.config.l_batch_size):
            ep_u, ep_v = batch_u[k], batch_v[k]
            u_idx = np.random.randint(0, len(ep_u))
            v_idx = np.random.randint(0, len(ep_v))
            u_ = ep_u[u_idx]
            v_ = ep_v[v_idx]
            u.append(u_)
            v.append(v_)
        u = tensor(u, dev)
        v = tensor(v, dev)

        f_u_, f_v_ = self.get_fuv(u, v)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.l_opt.zero_grad()
        total_loss.backward()
        self.l_opt.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        out = self.vectors(torch.cat([u, v]))
        f_u = out[:self.config.l_batch_size]
        f_v = out[self.config.l_batch_size:]
        return f_u, f_v


class LaplaceExcursionRepresentationAgent(Agent):
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

        self.l_replay = config.l_replay_fn()
        self.vectors = config.vector_fn()
        self.l_opt = config.l_opt_fn(self.vectors.parameters())
        self.trajectory = []

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1),
                                  p=self.tau_probs_norm, size=config.l_batch_size)
        self.num_episodes = 0

        self.total_loss = []
        self.attractive_loss = []
        self.repulsive_loss = []

        self.eval_number = 0
        self.random_excursion_prob = 0.1
        self.random_excursion = False

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False
            if np.random.rand() < self.random_excursion_prob or self.num_episodes < 10:
                self.random_excursion = True

        s = config.state_normalizer(self.state)
        if self.random_excursion: self.trajectory.append(s)

        phi_s = self.get_fs(tensor(s, config.device).unsqueeze(0))
        q_values = self.q_net(phi_s)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob() or self.random_excursion:
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])

        ns = config.state_normalizer(next_state)
        phi_ns = self.get_fs(tensor(ns, config.device).unsqueeze(0))
        entry = [phi_s, action, reward, phi_ns, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True
            if self.random_excursion:
                self.trajectory.append(config.state_normalizer(self.state))
                self.l_replay.feed_batch([np.array(self.trajectory)])
                self.trajectory = []
                self.random_excursion = False
            self.num_episodes += 1
            self.train_representation()
        self.q_replay.feed_batch([[state, action, reward, next_state, done]])

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

    def get_fs(self, s):
        fs = self.vectors(s).view(self.config.d).cpu().detach().numpy()
        return fs

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        phi = self.get_fs(state)
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
        super(LaplaceExcursionRepresentationAgent, self).eval_episodes()
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        with torch.no_grad():
            out = self.vectors(torch.cat([states_, goals_]))
        f_s = out[:len(states_)]
        f_g = out[len(states_):]

        f_s = f_s.unsqueeze(2)
        f_g = f_g.unsqueeze(2)

        fig, ax = plt.subplots(nrows=len(goals), ncols=1, figsize=(6, 6 * 4))
        for g_k in range(len(goals)):
            g = f_g[g_k]
            l2_vec = (f_s - g)**2
            l2_vec = torch.sum(l2_vec.squeeze(2), 1)
            distance = np.zeros((15, 15))
            for k, s in enumerate(states):
                x, y = s
                distance[x][y] = l2_vec[k].item()
            sns.heatmap(distance, ax=ax[g_k])
            ax[g_k].set_title('Goal: {}, {}'.format(goals[g_k][0], goals[g_k][1]))
        self.eval_number +=1
        plt.savefig(os.path.join(heatmap_dir, 'heatmap_{}.png'.format(self.eval_number)))
        plt.close()

    def train_representation(self):
        bs, d, to, dev, = self.config.l_batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.l_replay.sample_array()
        samples = self.sample_tau()
        u, v = [], []
        for k in range(self.config.l_batch_size):
            ep = batch[k]
            # if len(ep) < 2: continue
            u_idx = np.random.randint(0, len(ep) - 1)
            v_idx = min(u_idx + samples[k], len(ep) - 1)
            u_ = ep[u_idx]
            v_ = ep[v_idx]
            u.append(u_)
            v.append(v_)

        u = tensor(u, dev)
        v = tensor(v, dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        batch_u = self.l_replay.sample_array()
        batch_v = self.l_replay.sample_array()
        u, v = [], []
        for k in range(self.config.l_batch_size):
            ep_u, ep_v = batch_u[k], batch_v[k]
            u_idx = np.random.randint(0, len(ep_u))
            v_idx = np.random.randint(0, len(ep_v))
            u_ = ep_u[u_idx]
            v_ = ep_v[v_idx]
            u.append(u_)
            v.append(v_)
        u = tensor(u, dev)
        v = tensor(v, dev)

        f_u_, f_v_ = self.get_fuv(u, v)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.l_opt.zero_grad()
        total_loss.backward()
        self.l_opt.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        out = self.vectors(torch.cat([u, v]))
        f_u = out[:self.config.l_batch_size]
        f_v = out[self.config.l_batch_size:]
        return f_u, f_v