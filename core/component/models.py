from ..network import *
import torch
from torch import optim

import sys


class DeterministicDeltaModel:
    def __init__(self, cfg, device, state_channels, num_actions,
                 g_lr=5e-4, d_lr=5e-4, b_1=0.5, b_2=0.999, disc_net_class='DiscNet', n_d=5,
                 determ_net_class='DeterministicNet', round_for_disc=False, round_digits=-1):

        self.cfg = cfg

        self.state_channels = state_channels
        self.num_actions = num_actions

        determ_net_class = getattr(sys.modules[__name__], determ_net_class)
        self.deterministic_net = determ_net_class(state_channels, num_actions).to(device)
        self.mse_loss = nn.MSELoss()

        disc_class = getattr(sys.modules[__name__], disc_net_class)
        self.disc = disc_class(2 * state_channels, num_actions).to(device)

        self.opt_determ_net = optim.Adam(self.deterministic_net.parameters(), lr=g_lr, betas=(b_1, b_2))
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=d_lr, betas=(b_1, b_2))

        self.device = device

        # DiscNet Diagnostics
        self.ep_error_loss = []
        self.ep_true_errors = []
        self.ep_error_predictions = []

        self.determ_loss = []
        self.ep_mse = []

        self.pos = 0
        self.num_batches = None
        self.total_steps = 0
        self.n_d = n_d

        self.round_for_disc = round_for_disc
        self.round_digits = round_digits

        self.disc_loss = nn.SmoothL1Loss() if cfg.huber_loss else nn.MSELoss()

    def predict(self, states, action_labels):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        pred_next_states = states + self.deterministic_net.generate(states, actions)
        errors = self.disc.discriminate(states, actions, self.round(pred_next_states))
        return pred_next_states, errors
        # return pred_next_states

    def predict_nextstates(self, states, action_labels):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        pred_next_states = states + self.deterministic_net.generate(states, actions)
        return pred_next_states

    def predict_errors(self, states, action_labels, pred_next_states):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        errors = self.disc.discriminate(states, actions, self.round(pred_next_states))
        return errors

    def train(self, states, action_labels, next_states):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        deltas = self.deterministic_net.generate(states, actions)

        predictions = self.round(states + deltas)

        error_predictions = self.disc.discriminate(states, actions, predictions)
        true_errors = torch.reshape((next_states - predictions) ** 2, (-1, 10 * 10 * self.state_channels))
        true_errors = torch.mean(true_errors, dim=1)
        loss_disc = self.mse_loss(error_predictions, true_errors)

        self.opt_disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()

        loss_determ = self.mse_loss(deltas, next_states - states)
        self.opt_determ_net.zero_grad()
        loss_determ.backward()
        self.opt_determ_net.step()

        self.update_stats(loss_disc.squeeze().detach().item(),
                          true_errors.squeeze().detach().cpu().numpy(),
                          error_predictions.squeeze().detach().cpu().numpy(),
                          loss_determ.squeeze().detach().cpu().numpy())

        self.total_steps += 1

    def train_discriminator(self, states, action_labels, next_states):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        with torch.no_grad():
            deltas = self.deterministic_net.generate(states, actions)

        predictions = self.round(states + deltas)

        error_predictions = self.disc.discriminate(states, actions, predictions)
        true_errors = torch.mean((next_states - predictions) ** 2, (1, 2, 3))
        loss_disc = self.mse_loss(error_predictions, true_errors.unsqueeze(1)*100)
        self.opt_disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()

        loss_determ = self.mse_loss(deltas, next_states - states)

        self.update_stats(loss_disc.squeeze().detach().item(),
                          true_errors.squeeze().detach().cpu().numpy(),
                          error_predictions.squeeze().detach().cpu().numpy(),
                          loss_determ.squeeze().detach().cpu().numpy(),
                          true_errors.squeeze().detach().cpu().numpy())

        self.total_steps += 1

    def train_discriminator_direct(self, next_states, predictions, error_predictions):
        true_errors = torch.mean((next_states - predictions) ** 2, (1, 2, 3))

        if self.cfg.disc_clip:
            true_errors = torch.clamp(true_errors, min=self.cfg.threshold/self.cfg.clip_factor,
                                      max=self.cfg.threshold*self.cfg.clip_factor)

        loss_disc = self.disc_loss(error_predictions, true_errors.unsqueeze(1)*100)
        self.opt_disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()

        loss_determ = self.mse_loss(predictions, next_states)

        self.update_stats(loss_disc.squeeze().detach().item(),
                          true_errors.squeeze().detach().cpu().numpy(),
                          error_predictions.squeeze().detach().cpu().numpy()/100,
                          loss_determ.squeeze().detach().cpu().numpy(),
                          true_errors.squeeze().detach().cpu().numpy())

        self.total_steps += 1

    def train_model_direct(self, states, action_labels, next_states):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        deltas = self.deterministic_net.generate(states, actions)

        loss_determ = self.mse_loss(deltas, next_states - states)
        self.opt_determ_net.zero_grad()
        loss_determ.backward()
        self.opt_determ_net.step()

        return states + deltas

    def train_determ_model(self, states, action_labels, next_states):
        batch_size = states.shape[0]
        actions = torch.zeros((batch_size, self.num_actions)).to(self.device)
        actions[torch.arange(batch_size), action_labels] = 1
        deltas = self.deterministic_net.generate(states, actions)

        loss_determ = self.mse_loss(deltas, next_states - states)
        self.opt_determ_net.zero_grad()
        loss_determ.backward()
        self.opt_determ_net.step()

        self.update_stats_determ(loss_determ.squeeze().detach().cpu().numpy())

        self.total_steps += 1

    def round(self, predictions):
        if self.round_for_disc:
            predictions = torch.round(predictions * 10**self.round_digits)/(10**self.round_digits)
        return predictions

    def store_disc(self, store_path, training_step):
        torch.save(self.disc, os.path.join(store_path, "discriminator_{}".format(training_step)))

    def to_device(self, device):
        self.disc.to(device)
        self.deterministic_net.to(device)

    def write_tensorboard_determ(self, tb):
        mdl = np.mean(self.determ_loss)
        tb.add_scalar('/loss/model_mse', mdl, self.total_steps)
        return mdl

    def write_tensorboard(self, tb):
        true_errors, error_predictions = np.hstack(self.ep_true_errors), np.hstack(self.ep_error_predictions)
        percentiles = [10, 50, 90]
        for p in percentiles:
            t_p = np.percentile(true_errors, p)
            f_p = np.percentile(error_predictions, p)
            tb.add_scalar('/disc/percentile/{}_true_errors'.format(p), t_p, self.total_steps)
            tb.add_scalar('/disc/percentile/{}_error_predictions'
                          .format(p), f_p, self.total_steps)
        me = np.mean(self.ep_error_loss)
        tb.add_scalar('/disc/loss/error_loss', me, self.total_steps)
        mdl = np.mean(self.determ_loss)
        tb.add_scalar('/disc/loss/model_mse', mdl, self.total_steps)

        return me, mdl

    def update_stats_determ(self, loss_determ):
        if self.pos < self.num_batches:
            self.determ_loss.append(loss_determ)
        else:
            self.determ_loss[self.pos % self.num_batches] = loss_determ
        self.pos += 1

    def update_stats(self, loss, true_errors, error_predictions, loss_determ, mse=None):
        if self.pos < self.num_batches:
            self.ep_error_loss.append(loss)
            self.determ_loss.append(loss_determ)
            self.ep_true_errors.append(true_errors)
            self.ep_error_predictions.append(error_predictions)
            if mse is not None:
                self.ep_mse.append(mse)

        else:
            idx = self.pos % self.num_batches
            self.ep_error_loss[idx] = loss
            self.determ_loss[idx] = loss_determ
            self.ep_true_errors[idx] = true_errors
            self.ep_error_predictions[idx] = error_predictions
            if mse is not None:
                self.ep_mse[idx] = mse
        self.pos += 1

    def evaluate_beliefs(self, states, actions, next_states):
        one_hot_actions = torch.eye(self.num_actions).to(self.device)
        one_hot_actions = one_hot_actions[actions]
        beliefs = self.disc.discriminate(states, one_hot_actions, next_states)
        return beliefs

    def store(self, store_path, training_step):
        torch.save(self.deterministic_net, os.path.join(store_path, "deterministic_net_{}".format(training_step)))
        torch.save(self.disc, os.path.join(store_path, "discriminator_{}".format(training_step)))

    def load(self, ckpt_dir, suffix):
        self.deterministic_net = torch.load(os.path.join(ckpt_dir, "deterministic_net_{}".format(suffix)),
                                            map_location='cpu')
        # self.disc = torch.load(os.path.join(ckpt_dir, "discriminator_net_{}".format(suffix)), map_location='cpu')

    def load_disc(self, ckpt_dir, suffix):
        self.disc = torch.load(os.path.join(ckpt_dir, "discriminator_{}".format(suffix)), map_location='cpu')

    def load_model(self, ckpt_dir, suffix):
        self.deterministic_net = torch.load(os.path.join(ckpt_dir, "deterministic_net_{}".format(suffix)),
                                            map_location='cpu')

    def eval(self):
        self.deterministic_net.eval()
        self.disc.eval()


class RewardModel:
    def __init__(self, device, state_channels, num_actions, lr=5e-4):
        self.state_channels = state_channels
        self.num_actions = num_actions
        self.reward_net = SigmoidNet(state_channels, num_actions=num_actions).to(device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)
        self.err = []
        self.acc = []
        self.device = device
        self.criterion = nn.BCELoss()

    def train(self, states, actions, rewards):
        predictions = self.reward_net(states, actions)

        self.reward_net.zero_grad()
        loss = self.criterion(predictions, rewards.float())
        loss.backward()
        self.optimizer.step()
        self.err.append(loss.item())
        accuracy = torch.mean(((predictions > 0.5) == rewards.byte()).float())
        self.acc.append((accuracy, torch.sum(rewards), len(rewards)))

    def predict(self, states, actions):
        return (self.reward_net(states, actions) > 0.5).double()

    def store(self, store_path, training_step):
        torch.save(self.reward_net, os.path.join(store_path, "reward_model_{}".format(training_step)))

    def error(self):
        return self.acc[-1]

    def load(self, ckpt_dir, suffix):
        self.reward_net = torch.load(os.path.join(ckpt_dir, "reward_model_{}".format(suffix)), map_location='cpu')

    def eval(self):
        self.reward_net.eval()


class DoneModel:
    def __init__(self, device, state_channels, num_actions, lr=5e-4):
        self.state_channels = state_channels
        self.num_actions = num_actions
        self.done_net = SigmoidNet(state_channels, num_actions=num_actions).to(device)
        self.optimizer = optim.Adam(self.done_net.parameters(), lr=lr)
        self.err = []
        self.acc = []
        self.device = device
        self.criterion = nn.BCELoss()

    def train(self, states, actions, done):
        predictions = self.done_net(states, actions)

        self.criterion.weight = torch.ones_like(done).float().to(self.device) * 0.1
        self.criterion.weight[done.byte()] = 0.9

        self.done_net.zero_grad()
        loss = self.criterion(predictions, done.float())
        loss.backward()
        self.optimizer.step()
        self.err.append(loss.item())
        accuracy = torch.mean(((predictions > 0.5) == done.byte()).float())
        self.acc.append((accuracy, torch.sum(done), len(done)))

    def predict(self, states, actions):
        return self.done_net(states, actions) > 0.5

    def store(self, store_path, training_step):
        torch.save(self.done_net, os.path.join(store_path, "done_model_{}".format(training_step)))

    def error(self):
        return self.acc[-1]

    def load(self, ckpt_dir, suffix):
        self.done_net = torch.load(os.path.join(ckpt_dir, "done_model_{}".format(suffix)), map_location='cpu')

    def eval(self):
        self.done_net.eval()
