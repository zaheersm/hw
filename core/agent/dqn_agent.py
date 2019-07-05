from ..network import *
from ..component import *
from .base_agent import *

import matplotlib.pyplot as plt
import seaborn as sns


class DQNAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size, config.device)

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        q_values = self.network(config.state_normalizer(self.state))
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])
        entry = [self.state, action, reward, next_state, int(done), info]
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
        self.replay.feed_batch([[state, action, reward, next_state, done]])

        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        states = self.config.state_normalizer(states)
        next_states = self.config.state_normalizer(next_states)
        q_next = self.target_network(next_states) if config.use_target_network else self.network(next_states)
        q_next = q_next.detach().max(1)[0]
        terminals = tensor(terminals, self.config.device)
        rewards = tensor(rewards, self.config.device)
        q_next = self.config.discount * q_next * (1 - terminals).float()
        q_next.add_(rewards.float())
        actions = tensor(actions, self.config.device).long()
        q = self.network(states)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if config.use_target_network and self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action


class LaplaceAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []

        self.vectors = []
        self.vectors += [config.vector_fn() for _ in range(self.config.d)]
        parameters = []
        for vec in self.vectors:
            parameters += list(vec.parameters())
        self.optimizer = config.optimizer_fn(parameters)

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1), p=self.tau_probs_norm, size=config.batch_size)
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

        action = np.random.randint(0, config.action_dim)

        next_state, reward, done, info = self.task.step([action])
        entry = [self.state, action, reward, next_state, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.trajectory.append(state)

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

            self.replay.feed_batch([[self.trajectory]])
            self.trajectory = []

            self.num_episodes += 1

            if self.num_episodes > config.cold_start:
                self.train_representation()

    def train_representation(self):
        bs, d, to, dev, = self.config.batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.replay.sample()[0]

        samples = self.sample_tau()
        u = batch[np.arange(bs), to - 1 - samples]
        v = batch[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u, f_v = self.get_fuv(u, v)
        # loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # # Attractive terms
        # for k in range(self.config.d):
        #     loss += (f_u[k] - f_v[k]) ** 2
        # loss = 0.5*torch.mean(loss)
        f_u_ = torch.stack(f_u).squeeze(2).permute(1, 0)
        f_v_ = torch.stack(f_v).squeeze(2).permute(1, 0)

        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        samples = self.sample_tau()
        batch_u = self.replay.sample()[0]
        batch_v = self.replay.sample()[0]
        u = batch_u[np.arange(bs), to - 1 - samples]
        v = batch_v[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u, f_v = self.get_fuv(u, v)
        # orth_loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # for j in range(self.config.d):
        #     for k in range(self.config.d):
        #         delta = self.config.delta if j == k else 0
        #         orth_loss += (f_u[j]*f_u[k] - delta) * (f_v[j]*f_v[k] - delta)
        f_u_ = torch.stack(f_u).squeeze(2).permute(1, 0)
        f_v_ = torch.stack(f_v).squeeze(2).permute(1, 0)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        f_u, f_v = [], []
        for k in range(self.config.d):
            out = self.vectors[k](torch.cat([u, v]))
            f_u.append(out[:self.config.batch_size])
            f_v.append(out[self.config.batch_size:])
        return f_u, f_v

    def eval_step(self, state):
        action = np.random.randint(0, self.config.action_dim)
        return action

    def eval_episodes(self):
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        f_s = []
        f_g = []
        for k in range(self.config.d):
            with torch.no_grad():
                out = self.vectors[k](torch.cat([states_, goals_]))
            f_s.append(out[:len(states_)])
            f_g.append(out[len(states_):])

        f_s = torch.stack(f_s).permute(1, 0, 2)
        f_g = torch.stack(f_g).permute(1, 0, 2)

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

    def save(self):
        dir = self.config.get_eigvecdir()
        path = os.path.join(dir, "vec_{}")
        for k in range(self.config.d):
            torch.save(self.vectors[k].state_dict(), path.format(k))

    def load(self, dir):
        path = os.path.join(dir, "vec_{}")
        for k in range(self.config.d):
            self.vectors[k].load_state_dict(torch.load(path.format(k)))


class LaplaceReLUAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []

        self.vectors = []
        self.vectors += [config.vector_fn() for _ in range(self.config.d)]
        parameters = []
        for vec in self.vectors:
            parameters += list(vec.parameters())
        self.optimizer = config.optimizer_fn(parameters)

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1), p=self.tau_probs_norm, size=config.batch_size)
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

        action = np.random.randint(0, config.action_dim)

        next_state, reward, done, info = self.task.step([action])
        entry = [self.state, action, reward, next_state, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.trajectory.append(state)

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

            self.replay.feed_batch([[self.trajectory]])
            self.trajectory = []

            self.num_episodes += 1

            if self.num_episodes > config.cold_start:
                self.train_representation()

    def train_representation(self):
        bs, d, to, dev, = self.config.batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.replay.sample()[0]

        samples = self.sample_tau()
        u = batch[np.arange(bs), to - 1 - samples]
        v = batch[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u, f_v = self.get_fuv(u, v)
        # loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # # Attractive terms
        # for k in range(self.config.d):
        #     loss += (f_u[k] - f_v[k]) ** 2
        # loss = 0.5*torch.mean(loss)
        f_u_ = torch.stack(f_u).squeeze(2).permute(1, 0)
        f_v_ = torch.stack(f_v).squeeze(2).permute(1, 0)

        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        samples = self.sample_tau()
        batch_u = self.replay.sample()[0]
        batch_v = self.replay.sample()[0]
        u = batch_u[np.arange(bs), to - 1 - samples]
        v = batch_v[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u, f_v = self.get_fuv(u, v)
        # orth_loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # for j in range(self.config.d):
        #     for k in range(self.config.d):
        #         delta = self.config.delta if j == k else 0
        #         orth_loss += (f_u[j]*f_u[k] - delta) * (f_v[j]*f_v[k] - delta)
        f_u_ = torch.stack(f_u).squeeze(2).permute(1, 0)
        f_v_ = torch.stack(f_v).squeeze(2).permute(1, 0)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        f_u, f_v = [], []
        for k in range(self.config.d):
            out = torch.nn.functional.relu(self.vectors[k](torch.cat([u, v])))
            f_u.append(out[:self.config.batch_size])
            f_v.append(out[self.config.batch_size:])
        return f_u, f_v

    def eval_step(self, state):
        action = np.random.randint(0, self.config.action_dim)
        return action

    def eval_episodes(self):
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        f_s = []
        f_g = []
        for k in range(self.config.d):
            with torch.no_grad():
                out = torch.nn.functional.relu(self.vectors[k](torch.cat([states_, goals_])))
            f_s.append(out[:len(states_)])
            f_g.append(out[len(states_):])

        f_s = torch.stack(f_s).permute(1, 0, 2)
        f_g = torch.stack(f_g).permute(1, 0, 2)

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

        fig, ax = plt.subplots(nrows=self.config.d, ncols=1, figsize=(6, 6 * self.config.d))
        for d in range(self.config.d):
            values = np.zeros((15, 15))
            for k, s in enumerate(states):
                v = f_s[k][d].item()
                x, y = s
                values[x][y] = v
            sns.heatmap(values, ax=ax[d])
            ax[g_k].set_title('d: {}'.format(d))
        plt.savefig(os.path.join(heatmap_dir, 'components_{}.png'.format(self.eval_number)))
        plt.close()

    def save(self):
        dir = self.config.get_eigvecdir()
        path = os.path.join(dir, "vec_{}")
        for k in range(self.config.d):
            torch.save(self.vectors[k].state_dict(), path.format(k))

    def load(self, dir):
        path = os.path.join(dir, "vec_{}")
        for k in range(self.config.d):
            self.vectors[k].load_state_dict(torch.load(path.format(k)))


class LaplaceHeadAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []

        self.vectors = config.vector_fn()
        self.optimizer = config.optimizer_fn(self.vectors.parameters())

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1), p=self.tau_probs_norm, size=config.batch_size)
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

        action = np.random.randint(0, config.action_dim)

        next_state, reward, done, info = self.task.step([action])
        entry = [self.state, action, reward, next_state, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.trajectory.append(state)

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

            self.replay.feed_batch([[self.trajectory]])
            self.trajectory = []

            self.num_episodes += 1

            if self.num_episodes > config.cold_start:
                self.train_representation()

    def train_representation(self):
        bs, d, to, dev, = self.config.batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.replay.sample()[0]

        samples = self.sample_tau()
        u = batch[np.arange(bs), to - 1 - samples]
        v = batch[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        # loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # # Attractive terms
        # for k in range(self.config.d):
        #     loss += (f_u[k] - f_v[k]) ** 2
        # loss = 0.5*torch.mean(loss)
        # f_u_ = f_u.squeeze(2).permute(1, 0)
        # f_v_ = f_v.squeeze(2).permute(1, 0)

        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        samples = self.sample_tau()
        batch_u = self.replay.sample()[0]
        batch_v = self.replay.sample()[0]
        u = batch_u[np.arange(bs), to - 1 - samples]
        v = batch_v[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        # orth_loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # for j in range(self.config.d):
        #     for k in range(self.config.d):
        #         delta = self.config.delta if j == k else 0
        #         orth_loss += (f_u[j]*f_u[k] - delta) * (f_v[j]*f_v[k] - delta)
        # f_u_ = f_u.squeeze(2).permute(1, 0)
        # f_v_ = f_v.squeeze(2).permute(1, 0)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        out = self.vectors(torch.cat([u, v]))
        f_u = out[:self.config.batch_size]
        f_v = out[self.config.batch_size:]
        return f_u, f_v

    def eval_step(self, state):
        action = np.random.randint(0, self.config.action_dim)
        return action

    def eval_episodes(self):
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        f_s = []
        f_g = []
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

    def save(self):
        dir = self.config.get_eigvecdir()
        path = os.path.join(dir, "vec")
        torch.save(self.vectors.state_dict(), path)

    def load(self, dir):
        path = os.path.join(dir, "vec")
        self.vectors.load_state_dict(torch.load(path))


class LaplaceHeadReLUAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.trajectory = []

        self.vectors = config.vector_fn()
        self.optimizer = config.optimizer_fn(self.vectors.parameters())

        self.tau = list(range(1, self.timeout+1))
        self.tau_probs = [config.lmbda**(x-1)-config.lmbda**x for x in self.tau]
        self.tau_probs_norm = [x/np.sum(self.tau_probs) for x in self.tau_probs]
        self.sample_tau = lambda: np.random.choice(range(1, config.timeout+1), p=self.tau_probs_norm, size=config.batch_size)
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

        action = np.random.randint(0, config.action_dim)

        next_state, reward, done, info = self.task.step([action])
        entry = [self.state, action, reward, next_state, int(done), info]
        self.state = next_state

        state, action, reward, next_state, done, _ = entry

        self.trajectory.append(state)

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

            self.replay.feed_batch([[self.trajectory]])
            self.trajectory = []

            self.num_episodes += 1

            if self.num_episodes > config.cold_start:
                self.train_representation()

    def train_representation(self):
        bs, d, to, dev, = self.config.batch_size, self.config.d, self.config.timeout, self.config.device
        delta, beta = self.config.delta, self.config.beta

        batch = self.replay.sample()[0]

        samples = self.sample_tau()
        u = batch[np.arange(bs), to - 1 - samples]
        v = batch[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        # loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # # Attractive terms
        # for k in range(self.config.d):
        #     loss += (f_u[k] - f_v[k]) ** 2
        # loss = 0.5*torch.mean(loss)
        # f_u_ = f_u.squeeze(2).permute(1, 0)
        # f_v_ = f_v.squeeze(2).permute(1, 0)

        loss = 0.5*torch.mean(torch.sum((f_u_ - f_v_)**2, 1))

        samples = self.sample_tau()
        batch_u = self.replay.sample()[0]
        batch_v = self.replay.sample()[0]
        u = batch_u[np.arange(bs), to - 1 - samples]
        v = batch_v[np.arange(bs), np.ones(bs).astype(np.int64) * to - 1]
        u = tensor(self.config.state_normalizer(u), dev)
        v = tensor(self.config.state_normalizer(v), dev)

        f_u_, f_v_ = self.get_fuv(u, v)
        # orth_loss = tensor(np.zeros((self.config.batch_size, 1)), self.config.device)
        # for j in range(self.config.d):
        #     for k in range(self.config.d):
        #         delta = self.config.delta if j == k else 0
        #         orth_loss += (f_u[j]*f_u[k] - delta) * (f_v[j]*f_v[k] - delta)
        # f_u_ = f_u.squeeze(2).permute(1, 0)
        # f_v_ = f_v.squeeze(2).permute(1, 0)

        dot_product = torch.bmm(f_u_.view(bs, 1, d), f_v_.view(bs, d, 1))**2
        dot_product = dot_product.view(bs)

        orth_loss = dot_product - delta*torch.sum(f_u_ ** 2, 1) - delta*torch.sum(f_v_ ** 2, 1) + d*delta**2
        orth_loss = beta*torch.mean(orth_loss)
        total_loss = loss + orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss.append(total_loss.item())
        self.attractive_loss.append(loss.item())
        self.repulsive_loss.append(orth_loss.item())

    def get_fuv(self, u, v):
        out = torch.nn.functional.relu((self.vectors(torch.cat([u, v]))))
        f_u = out[:self.config.batch_size]
        f_v = out[self.config.batch_size:]
        return f_u, f_v

    def eval_step(self, state):
        action = np.random.randint(0, self.config.action_dim)
        return action

    def eval_episodes(self):
        # This function is called eval_episodes for legacy reasons
        heatmap_dir = self.config.get_heatmapdir()
        states = self.task.get_eval_states()
        goals = self.task.get_eval_goal_states()

        states_ = tensor(self.config.state_normalizer(states), self.config.device)
        goals_ = tensor(self.config.state_normalizer(goals), self.config.device)

        f_s = []
        f_g = []
        with torch.no_grad():
            out = torch.nn.functional.relu(self.vectors(torch.cat([states_, goals_])))
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

        fig, ax = plt.subplots(nrows=self.config.d, ncols=1, figsize=(6, 6 * self.config.d))
        for d in range(min(self.config.d, 50)):
            values = np.zeros((15, 15))
            for k, s in enumerate(states):
                v = f_s[k][d].item()
                x, y = s
                values[x][y] = v
            sns.heatmap(values, ax=ax[d])
            ax[g_k].set_title('d: {}'.format(d))
        plt.savefig(os.path.join(heatmap_dir, 'components_{}.png'.format(self.eval_number)))
        plt.close()

    def save(self):
        dir = self.config.get_eigvecdir()
        path = os.path.join(dir, "vec")
        torch.save(self.vectors.state_dict(), path)

    def load(self, dir):
        path = os.path.join(dir, "vec")
        self.vectors.load_state_dict(torch.load(path))


class DQNLaplaceAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size, config.device)

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.vectors = []
        self.vectors += [config.vector_fn() for _ in range(self.config.d)]

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        s = config.state_normalizer(self.state)

        phi_s = self.get_fs(tensor(s, config.device).unsqueeze(0))
        q_values = self.network(phi_s)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])

        ns = config.state_normalizer(next_state)
        phi_ns = self.get_fs(tensor(ns, config.device).unsqueeze(0))
        # phi_s, phi_ns = self.get_fsns(tensor(self.state, config.device).unsqueeze(0), tensor(next_state, config.device).unsqueeze(0))
        # phi_s = torch.stack(phi_s).view(config.d).cpu().detach().numpy()
        # phi_ns = torch.stack(phi_ns).view(config.d).cpu().detach().numpy()

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
        self.replay.feed_batch([[state, action, reward, next_state, done]])

        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        # states = self.config.state_normalizer(states)
        # next_states = self.config.state_normalizer(next_states)
        q_next = self.target_network(next_states) if config.use_target_network else self.network(next_states)
        q_next = q_next.detach().max(1)[0]
        terminals = tensor(terminals, self.config.device)
        rewards = tensor(rewards, self.config.device)
        q_next = self.config.discount * q_next * (1 - terminals).float()
        q_next.add_(rewards.float())
        actions = tensor(actions, self.config.device).long()
        q = self.network(states)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if config.use_target_network and self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def get_fsns(self, s, ns):
        f_u, f_v = [], []
        for k in range(self.config.d):
            out = self.vectors[k](torch.cat([s, ns]))
            k_u, k_v = out
            f_u.append(k_u)
            f_v.append(k_v)
        f_u = torch.stack(f_u).view(self.config.d).cpu().detach().numpy()
        f_v = torch.stack(f_v).view(self.config.d).cpu().detach().numpy()
        return f_u, f_v

    def get_fs(self, s):
        f_u = []
        for k in range(self.config.d):
            out = self.vectors[k](s)
            f_u.append(out)
        f_u = torch.stack(f_u).view(self.config.d).cpu().detach().numpy()
        return f_u

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        phi = self.get_fs(state)
        q = self.network(phi)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def load(self, dir):
        path = os.path.join(dir, "vec_{}")
        for k in range(self.config.d):
            self.vectors[k].load_state_dict(torch.load(path.format(k)))


class DQNLaplaceHeadAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size, config.device)

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.vectors = config.vector_fn()

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        s = config.state_normalizer(self.state)

        phi_s = self.get_fs(tensor(s, config.device).unsqueeze(0))
        q_values = self.network(phi_s)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])

        ns = config.state_normalizer(next_state)
        phi_ns = self.get_fs(tensor(ns, config.device).unsqueeze(0))
        # phi_s, phi_ns = self.get_fsns(tensor(self.state, config.device).unsqueeze(0), tensor(next_state, config.device).unsqueeze(0))
        # phi_s = torch.stack(phi_s).view(config.d).cpu().detach().numpy()
        # phi_ns = torch.stack(phi_ns).view(config.d).cpu().detach().numpy()

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
        self.replay.feed_batch([[state, action, reward, next_state, done]])

        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        # states = self.config.state_normalizer(states)
        # next_states = self.config.state_normalizer(next_states)
        q_next = self.target_network(next_states) if config.use_target_network else self.network(next_states)
        q_next = q_next.detach().max(1)[0]
        terminals = tensor(terminals, self.config.device)
        rewards = tensor(rewards, self.config.device)
        q_next = self.config.discount * q_next * (1 - terminals).float()
        q_next.add_(rewards.float())
        actions = tensor(actions, self.config.device).long()
        q = self.network(states)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if config.use_target_network and self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    # def get_fsns(self, s, ns):
    #     f_u, f_v = [], []
    #     for k in range(self.config.d):
    #         out = self.vectors[k](torch.cat([s, ns]))
    #         k_u, k_v = out
    #         f_u.append(k_u)
    #         f_v.append(k_v)
    #     f_u = torch.stack(f_u).view(self.config.d).cpu().detach().numpy()
    #     f_v = torch.stack(f_v).view(self.config.d).cpu().detach().numpy()
    #     return f_u, f_v

    # def get_fs(self, s):
    #     f_u = []
    #     for k in range(self.config.d):
    #         out = self.vectors[k](s)
    #         f_u.append(out)
    #     f_u = torch.stack(f_u).view(self.config.d).cpu().detach().numpy()
    #     return f_u

    def get_fs(self, s):
        fs = self.vectors(s).view(self.config.d).cpu().detach().numpy()
        return fs

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        phi = self.get_fs(state)
        q = self.network(phi)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def load(self, dir):
        path = os.path.join(dir, "vec")
        self.vectors.load_state_dict(torch.load(path))


class DQNLaplaceModulationAgent(Agent):
    def __init__(self, config):
        Agent.__init__(self, config)

        self.replay = config.replay_fn()

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size, config.device)

        self.reset = True
        self.ep_steps = 0

        self.timeout = config.timeout
        self.task = config.task_fn()

        self.state = None
        self.action = None
        self.next_state = None

        self.vectors = config.vector_fn()
        # self.prototypes = self.task.get_prototypes()
        self.prototypes = None
        self.proto_embeddings = None

    def load_prototypes(self):
        self.prototypes = self.config.state_normalizer(tensor(self.task.get_prototypes(), self.config.device))
        with torch.no_grad():
            self.proto_embeddings = self.vectors(self.prototypes)

    def step(self):
        config = self.config

        if self.reset is True:
            self.state = self.task.reset()
            self.reset = False

        s = config.state_normalizer(self.state)
        phi_s = self.get_fs(tensor(s, config.device).unsqueeze(0))
        q_values = self.network(s, tensor(phi_s, self.config.device).unsqueeze(0), self.proto_embeddings).squeeze(0)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self.task.step([action])

        ns = config.state_normalizer(next_state)
        phi_ns = self.get_fs(tensor(ns, config.device).unsqueeze(0))

        entry = [s, phi_s, action, reward, ns, phi_ns, int(done), info]
        self.state = next_state

        state, phi_s, action, reward, next_state, phi_ns, done, _ = entry

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        reward = config.reward_normalizer(reward)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True
        self.replay.feed_batch([[state, phi_s, action, reward, next_state, phi_ns, done]])

        experiences = self.replay.sample()
        states, phi_s, actions, rewards, next_states, phi_ns, terminals = experiences
        q_next = self.target_network(next_states, phi_ns, self.proto_embeddings) if config.use_target_network else self.network(next_states, phi_ns, self.proto_embeddings)
        q_next = q_next.detach().max(1)[0]
        terminals = tensor(terminals, self.config.device)
        rewards = tensor(rewards, self.config.device)
        q_next = self.config.discount * q_next * (1 - terminals).float()
        q_next.add_(rewards.float())
        actions = tensor(actions, self.config.device).long()
        q = self.network(states, phi_s, self.proto_embeddings)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if config.use_target_network and self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def get_fs(self, s):
        with torch.no_grad():
            fs = self.vectors(s).view(self.config.d).cpu().detach().numpy()
        return fs

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        phi = tensor(self.get_fs(state), self.config.device)
        q = self.network(state, phi.unsqueeze(0), self.proto_embeddings)[0]
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def load(self, dir):
        path = os.path.join(dir, "vec")
        self.vectors.load_state_dict(torch.load(path))