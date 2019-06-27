from ..network import *
from .replay import Replay


class PlannerOracle:
    def __init__(self, cfg):
        self.cfg = cfg
        self.replay = cfg.replay_fn()
        self.model = cfg.model
        self.model.model.deterministic_net.to(cfg.device)
        self.step = 0
        self.thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 5e-5, 1e-5, cfg.threshold]

        self.ratio_reject = [[] for _ in self.thresholds]
        self.ratio_reject_oracle = [[] for _ in self.thresholds]

        self.reward_acc = []
        self.done_acc = []

        self.buffer = []

        if cfg.save_error_dist:
            self.error_memory = Replay(cfg.error_mem_size, cfg.batch_size)

    def feed_batch(self, transition):
        self.buffer.append(transition[0])
        if len(self.buffer) % 32 == 0:
            buffer = list(map(lambda x: np.asarray(x), zip(*self.buffer)))
            states, actions, rewards, next_states, dones = self.to_tensor(buffer)
            with torch.no_grad():
                predicted_dones = self.model.predict_done(self.shape_image(states), actions)
                self.update_done_acc(dones, predicted_dones)
                predicted_rewards = self.model.predict_reward(self.shape_image(states), actions)
                self.update_reward_acc(rewards, predicted_rewards)

            states_nonterm = states[~predicted_dones]
            actions_nonterm = actions[~predicted_dones]
            next_states_nonterm = next_states[~predicted_dones]
            with torch.no_grad():
                states_nonterm = self.shape_image(states_nonterm)
                predictions, beliefs = self.model.predict_nextstate_errors(states_nonterm.to(self.cfg.device),
                                                                           actions_nonterm.to(self.cfg.device))
            predictions = self.shape_flat(predictions)
            mse_error = ((predictions - next_states_nonterm)**2).mean(dim=1)
            self.update_reject_stats_oracle(mse_error)
            filtered_predictions = self.filter(next_states, predictions, predicted_dones, mse_error)
            transitions = [states, actions, predicted_rewards, filtered_predictions, predicted_dones]
            self.buffer = []
            self.replay.feed_tensors(transitions)

            if self.cfg.save_error_dist:
                self.error_memory.feed(mse_error.detach().cpu().numpy())

    def sample(self):
        return self.replay.sample()

    def filter(self, next_states, predictions, dones, error):
        filtered_predictions = next_states.clone()
        no_plan = dones.clone()
        if self.cfg.selective_planning:
            cond = (error > self.cfg.threshold)
            no_plan[~dones] = cond
            filtered_predictions[~no_plan] = predictions[~cond]
        else:
            filtered_predictions[~no_plan] = predictions
        return filtered_predictions

    def to_tensor(self, transitions):
        states, actions, rewards, next_states, dones = transitions
        states = torch.from_numpy(states).float().to(self.cfg.device)
        actions = torch.from_numpy(actions).long().to(self.cfg.device)
        rewards = torch.from_numpy(rewards).double().to(self.cfg.device)
        next_states = torch.from_numpy(next_states).float().to(self.cfg.device)
        dones = torch.from_numpy(dones).byte().to(self.cfg.device)
        return states, actions, rewards, next_states, dones

    def update_reject_stats(self, error):
        for k, t in enumerate(self.thresholds):
            self.ratio_reject[k].append(torch.sum(error > t).float().item()/len(error))

    def update_reject_stats_oracle(self, error):
        for k, t in enumerate(self.thresholds):
            self.ratio_reject_oracle[k].append(torch.sum(error > t).float().item()/len(error))

    def reset_reject_stats(self):
        for k, t in enumerate(self.thresholds):
            self.ratio_reject[k] = []

    def reset_reject_stats_oracle(self):
        for k, t in enumerate(self.thresholds):
            self.ratio_reject_oracle[k] = []

    def get_statistics(self):
        stats = [np.mean(x) for x in self.ratio_reject]
        self.reset_reject_stats()
        return stats

    def get_statistics_oracle(self):
        stats = [np.mean(x) for x in self.ratio_reject_oracle]
        self.reset_reject_stats_oracle()
        return stats

    def update_done_acc(self, dones, predictions):
        self.done_acc.append(torch.mean((predictions == dones).float()).item())

    def get_done_acc(self):
        return np.mean(self.done_acc)

    def update_reward_acc(self, rewards, predictions):
        self.reward_acc.append(torch.mean((predictions.byte() == rewards.byte()).float()).item())

    def get_reward_acc(self):
        return np.mean(self.reward_acc)

    def shape_image(self, x):
        return x.reshape(-1, 10, 10, self.cfg.state_dim[-1]).permute(0, 3, 1, 2)

    def shape_flat(self, x):
        return x.permute(0, 2, 3, 1).reshape(-1, np.prod(x.size()[1:]))

    def save_errors(self, path, step):
        errors = np.hstack(self.error_memory.data)
        np.save(os.path.join(path, 'step_{}.npy'.format(step)), errors)


class PlannerLearnedSelect(PlannerOracle):
    def __init__(self, cfg):
        PlannerOracle.__init__(self, cfg)

    def feed_batch(self, transition):
        self.buffer.append(transition[0])
        if len(self.buffer) % 32 == 0:
            buffer = list(map(lambda x: np.asarray(x), zip(*self.buffer)))
            states, actions, rewards, next_states, dones = self.to_tensor(buffer)
            with torch.no_grad():
                predicted_dones = self.model.predict_done(self.shape_image(states), actions)
                self.update_done_acc(dones, predicted_dones)
                predicted_rewards = self.model.predict_reward(self.shape_image(states), actions)
                self.update_reward_acc(rewards, predicted_rewards)

            states_nonterm = states[~predicted_dones]
            actions_nonterm = actions[~predicted_dones]
            next_states_nonterm = next_states[~predicted_dones]
            with torch.no_grad():
                states_nonterm = self.shape_image(states_nonterm)
                predictions, beliefs = self.model.predict_nextstate_errors(states_nonterm.to(self.cfg.device),
                                                                           actions_nonterm.to(self.cfg.device))
                beliefs = beliefs.squeeze(1)/100
            predictions = self.shape_flat(predictions)

            self.update_reject_stats(beliefs)
            mse_error = ((predictions - next_states_nonterm)**2).mean(dim=1)
            self.update_reject_stats_oracle(mse_error)

            filtered_predictions = self.filter(next_states, predictions, predicted_dones, beliefs)
            transitions = [states, actions, predicted_rewards, filtered_predictions, predicted_dones]
            self.buffer = []
            self.replay.feed_tensors(transitions)

            if self.cfg.save_error_dist:
                self.error_memory.feed(mse_error.detach().cpu().numpy())


class PlannerOnlineErrorEstimation(PlannerLearnedSelect):
    def __init__(self, cfg):
        PlannerLearnedSelect.__init__(self, cfg)
        self.model.model.num_batches = 6250

    # def learn_error(self):
    #     batch = self.replay.sample()

    def feed_batch(self, transition):
        self.replay.feed_batch(transition)

    def feed_batch(self, transition):
        self.buffer.append(transition[0])
        if len(self.buffer) % 32 == 0:
            buffer = list(map(lambda x: np.asarray(x), zip(*self.buffer)))
            states, actions, rewards, next_states, dones = self.to_tensor(buffer)

            with torch.no_grad():
                predictions = self.model.predict_nextstates(self.shape_image(states).to(self.cfg.device),
                                                            actions.to(self.cfg.device))
            predictions = self.shape_flat(predictions)
            transitions = [states, actions, rewards, next_states, dones, predictions]
            self.buffer = []
            self.replay.feed_tensors(transitions)

    def sample(self):
        transitions = self.replay.sample()
        states, actions, rewards, next_states, dones, predictions = transitions

        states_nt = states[~dones]
        actions_nt = actions[~dones]
        next_states_nt = next_states[~dones]
        predictions_nt = predictions[~dones]
        states_nt = self.shape_image(states_nt)
        predictions_nt = self.shape_image(predictions_nt)

        if self.step % self.cfg.disc_train_freq == 0:
            errors = self.model.predict_errors(states_nt.to(self.cfg.device),
                                               actions_nt.to(self.cfg.device),
                                               predictions_nt.to(self.cfg.device))

            self.model.model.train_discriminator_direct(self.shape_image(next_states_nt), predictions_nt, errors)
            if self.step % self.cfg.disc_log_freq == 0: self.model.model.write_tensorboard(self.cfg.tensorboard)
        else:
            with torch.no_grad():
                errors = self.model.predict_errors(states_nt.to(self.cfg.device),
                                                   actions_nt.to(self.cfg.device),
                                                   predictions_nt.to(self.cfg.device))

        predictions_nt = self.shape_flat(predictions_nt)
        errors = errors/100
        self.update_reject_stats(errors)
        mse_error = ((predictions_nt - next_states_nt)**2).mean(dim=1)
        self.update_reject_stats_oracle(mse_error)
        filtered_predictions = self.filter(next_states, predictions_nt, dones, errors.squeeze())

        self.step += 1

        return [states, actions, rewards, filtered_predictions, dones]


class PlannerOnlineModelErrorEstimation(PlannerLearnedSelect):
    def __init__(self, cfg):
        PlannerLearnedSelect.__init__(self, cfg)
        self.model.model.num_batches = 6250

    def feed_batch(self, transition):
        self.replay.feed_batch(transition)

    def sample(self):
        transitions = self.replay.sample()
        states, actions, rewards, next_states, dones = transitions

        states_nt = self.shape_image(states[~dones]).to(self.cfg.device)
        actions_nt = actions.to(self.cfg.device)[~dones]
        next_states_nt = self.shape_image(next_states[~dones]).to(self.cfg.device)

        if self.step % self.cfg.model_train_freq == 0:
            predictions_nt = self.model.model.train_model_direct(states_nt, actions_nt, next_states_nt).detach()
        else:
            with torch.no_grad():
                predictions_nt = self.model.predict_nextstates(states_nt, actions_nt)

        if self.step % self.cfg.disc_train_freq == 0:
            errors = self.model.predict_errors(states_nt, actions_nt, predictions_nt)

            self.model.model.train_discriminator_direct(next_states_nt, predictions_nt, errors)
            if self.step % self.cfg.disc_log_freq == 0: self.model.model.write_tensorboard(self.cfg.tensorboard)
        else:
            with torch.no_grad():
                errors = self.model.predict_errors(states_nt.to(self.cfg.device),
                                                   actions_nt.to(self.cfg.device),
                                                   predictions_nt.to(self.cfg.device))
        predictions_nt = self.shape_flat(predictions_nt)
        next_states_nt = self.shape_flat(next_states_nt)
        errors = errors/100
        self.update_reject_stats(errors)
        mse_error = ((predictions_nt - next_states_nt)**2).mean(dim=1)
        self.update_reject_stats_oracle(mse_error)
        filtered_predictions = self.filter(next_states, predictions_nt, dones, errors.squeeze())

        self.step += 1

        return [states, actions, rewards, filtered_predictions, dones]


class PlannerTrueModel:
    # Root node planning
    def __init__(self, cfg):
        self.cfg = cfg
        self.replay = cfg.replay_fn()
        self.true_model = cfg.true_model

        # The agent should set up the task
        self.task = None
        self.state = None

    def reset(self):
        self.state = self.task.reset()
        return self.state

    def step(self, action):
        raw_state = self.true_model.extract_state(self.task)
        next_state, reward, done, info = self.task.step([action])
        self.replay.feed_batch([[self.state, action, reward, next_state, int(done), raw_state]])
        self.state = next_state
        return next_state, reward, done, info

    def sample(self):
        states, actions, rewards, next_states, dones, raw_states = self.replay.sample()
        self.true_model.reset_states(raw_states)
        sim_next_states, sim_rewards, sim_dones, rollout_lengths = self.true_model.step(actions, np.zeros_like(next_states),
                                                                   np.zeros_like(rewards), np.zeros_like(dones),
                                                                   self.cfg.rollout_length)
        return self.to_tensor([states, actions, sim_rewards, sim_next_states, sim_dones, rollout_lengths])

    def to_tensor(self, transitions):
        states, actions, rewards, next_states, dones, rollout_lengths = transitions
        states = torch.from_numpy(states).float().to(self.cfg.device)
        actions = torch.from_numpy(actions).long().to(self.cfg.device)
        rewards = torch.from_numpy(rewards).double().to(self.cfg.device)
        next_states = torch.from_numpy(next_states).float().to(self.cfg.device)
        dones = torch.from_numpy(dones).byte().to(self.cfg.device)
        rollout_lengths = torch.from_numpy(rollout_lengths).float().to(self.cfg.device)
        return states, actions, rewards, next_states, dones, rollout_lengths


class PlannerTrueModelER:
    # Planner for Zach's Strategy
    def __init__(self, cfg):
        self.cfg = cfg
        self.replay = cfg.replay_fn()
        self.true_model = cfg.true_model

        # The agent should set up the task
        self.task = None
        self.state = None
        self.num_rollouts = cfg.num_rollouts

    def reset(self):
        self.state = self.task.reset()
        return self.state

    def step(self, action):
        raw_state = self.true_model.extract_state(self.task)
        next_state, reward, done, info = self.task.step([action])
        self.replay.feed_batch([[self.state, action, reward, next_state, int(done), raw_state]])
        self.state = next_state
        return next_state, reward, done, info

    def sample(self):

        for _ in range(self.num_rollouts):
            states, _, _, _, _, raw_states = self.replay.sample()
            self.true_model.reset_states(raw_states)
            self.true_model.add_experience(self.replay, states, self.cfg.rollout_length)

        states, actions, rewards, next_states, dones, _ = self.replay.sample()
        return self.to_tensor([states, actions, rewards, next_states, dones, np.ones_like(rewards)])

    def to_tensor(self, transitions):
        states, actions, rewards, next_states, dones, rollout_lengths = transitions
        states = torch.from_numpy(states).float().to(self.cfg.device)
        actions = torch.from_numpy(actions).long().to(self.cfg.device)
        rewards = torch.from_numpy(rewards).double().to(self.cfg.device)
        next_states = torch.from_numpy(next_states).float().to(self.cfg.device)
        dones = torch.from_numpy(dones).byte().to(self.cfg.device)
        rollout_lengths = torch.from_numpy(rollout_lengths).float().to(self.cfg.device)
        return states, actions, rewards, next_states, dones, rollout_lengths


class PlannerTrueModelSeparateBuffers:
    # We'll use Zach's strategy but we'll keep a separate buffer for simulated experience
    # This will allow us to reject a transition if e > t, and instead sample real experience
    # to keep the number of transitions equal
    def __init__(self, cfg):
        self.cfg = cfg
        self.replay = cfg.replay_fn()
        self.sim_replay = cfg.sim_replay_fn()

        self.true_model = cfg.true_model

        # The agent should set up the task
        self.task = None
        self.state = None

    def reset(self):
        self.state = self.task.reset()
        return self.state

    def step(self, action):
        next_state, reward, done, info = self.task.step([action])
        raw_state = self.true_model.extract_state(self.task)
        self.replay.feed_batch([[self.state, action, reward, next_state, int(done), raw_state]])
        self.state = next_state
        return next_state, reward, done, info

    def sample(self):

        _, _, _, next_states, dones, raw_states = self.replay.sample()

        # Only simulating forward from the non-terminal states
        idx = np.where(dones != 1)[0]
        raw_states = raw_states[idx]
        next_states = next_states[idx]
        self.true_model.reset_states(raw_states)
        batch = self.true_model.add_experience_v3(self.sim_replay, next_states, self.cfg.rollout_length)
        if len(batch) > 0: self.replay.feed_batch(batch)

        sim_states, sim_actions, sim_rewards, sim_next_states, sim_dones, _ = self.sim_replay.sample(self.cfg.sim_batch_size)
        states, actions, rewards, next_states, dones, _ = self.replay.sample(self.cfg.batch_size - len(sim_states))

        states = np.concatenate((states, sim_states))
        actions = np.concatenate((actions, sim_actions))
        rewards = np.concatenate((rewards, sim_rewards))
        next_states = np.concatenate((next_states, sim_next_states))
        dones = np.concatenate((dones, sim_dones))

        return self.to_tensor([states, actions, rewards, next_states, dones, np.ones_like(rewards)])

    def to_tensor(self, transitions):
        states, actions, rewards, next_states, dones, rollout_lengths = transitions
        states = torch.from_numpy(states).float().to(self.cfg.device)
        actions = torch.from_numpy(actions).long().to(self.cfg.device)
        rewards = torch.from_numpy(rewards).double().to(self.cfg.device)
        next_states = torch.from_numpy(next_states).float().to(self.cfg.device)
        dones = torch.from_numpy(dones).byte().to(self.cfg.device)
        rollout_lengths = torch.from_numpy(rollout_lengths).float().to(self.cfg.device)
        return states, actions, rewards, next_states, dones, rollout_lengths