from .models import *


class DeterministicDeltaModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        num_input_channels = self.cfg.state_dim[-1]
        cfg.d_lr = cfg.g_lr if cfg.same_lr else cfg.d_lr
        self.model = DeterministicDeltaModel(cfg, cfg.device, num_input_channels, cfg.action_dim,
                                             g_lr=cfg.g_lr, d_lr=cfg.d_lr, b_1=cfg.b_1, b_2=cfg.b_2,
                                             disc_net_class=cfg.disc_net_class,
                                             determ_net_class=cfg.determ_net_class,
                                             round_for_disc=cfg.round_for_disc,
                                             round_digits=cfg.round_digits)
        self.reward_model = RewardModel(cfg.device, num_input_channels, cfg.action_dim,
                                        lr=cfg.r_lr)
        # Using the same learning rate as the reward model for the done model
        self.done_model = DoneModel(cfg.device, num_input_channels, cfg.action_dim,
                                    lr=cfg.r_lr)
        self.total_steps = 0

    def train_step(self, batch):

        bs, ba, bd = batch['state'], batch['action'], batch['done']
        self.done_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), bd.to(self.cfg.device))

        bs, ba, br, bns = batch['state'][~batch['done']], batch['action'][~batch['done']],\
                          batch['reward'][~batch['done']], batch['next_state'][~batch['done']]
        self.model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), bns.to(self.cfg.device))
        self.reward_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), br.to(self.cfg.device))
        self.total_steps += 1

    # def train_step_discriminator(self, batch):
    #     bs, ba, bd = batch['state'], batch['action'], batch['done']
    #     self.done_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), bd.to(self.cfg.device))
    #
    #     bs, ba, br, bns = batch['state'][~batch['done']], batch['action'][~batch['done']],\
    #                       batch['reward'][~batch['done']], batch['next_state'][~batch['done']]
    #     self.model.train_discriminator(bs.to(self.cfg.device), ba.to(self.cfg.device), bns.to(self.cfg.device))
    #     self.reward_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), br.to(self.cfg.device))
    #     self.total_steps += 1

    def train_step_disc_only(self, batch):
        bs, ba, br, bns = batch['state'][~batch['done']], batch['action'][~batch['done']],\
                          batch['reward'][~batch['done']], batch['next_state'][~batch['done']]
        self.model.train_discriminator(bs.to(self.cfg.device), ba.to(self.cfg.device), bns.to(self.cfg.device))
        self.total_steps += 1

    def train_step_determ_model(self, batch):
        bs, ba, br, bns = batch['state'][~batch['done']], batch['action'][~batch['done']],\
                          batch['reward'][~batch['done']], batch['next_state'][~batch['done']]
        self.model.train_determ_model(bs.to(self.cfg.device), ba.to(self.cfg.device), bns.to(self.cfg.device))
        self.total_steps += 1

    def train_reward_done_model(self, batch):
        bs, ba, bd = batch['state'], batch['action'], batch['done']
        self.done_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), bd.to(self.cfg.device))

        bs, ba, br, bns = batch['state'][~batch['done']], batch['action'][~batch['done']],\
                          batch['reward'][~batch['done']], batch['next_state'][~batch['done']]
        self.reward_model.train(bs.to(self.cfg.device), ba.to(self.cfg.device), br.to(self.cfg.device))

    def store(self, training_step):
        path = self.cfg.get_logdir()
        self.model.store(path, training_step)
        self.reward_model.store(path, training_step)
        self.done_model.store(path, training_step)

    def store_disc(self, training_step):
        path = self.cfg.get_logdir()
        self.model.store_disc(path, training_step)

    def running_true_false_diff(self):
        return self.gan.running_true[-1] - self.gan.running_false[-1]

    def running_true(self):
        return self.gan.running_true[-1]

    def running_false(self):
        return self.gan.running_false[-1]

    def reward_err(self):
        return self.reward_model.error()

    def done_err(self):
        return self.done_model.error()

    def load(self, suffix, ckpt_dir=None):
        ckpt_dir = self.cfg.get_logdir() if ckpt_dir is None else ckpt_dir
        self.model.load(ckpt_dir, suffix)
        self.reward_model.load(ckpt_dir, suffix)
        self.done_model.load(ckpt_dir, suffix)

    def load_disc(self, suffix, ckpt_dir=None):
        ckpt_dir = self.cfg.get_logdir() if ckpt_dir is None else ckpt_dir
        self.model.load_disc(ckpt_dir, suffix)

    def load_model(self, suffix, ckpt_dir=None):
        ckpt_dir = self.cfg.get_logdir() if ckpt_dir is None else ckpt_dir
        self.model.load_model(ckpt_dir, suffix)

    def eval(self):
        self.model.eval()
        self.reward_model.eval()

    def predict_nextstate_errors(self, states, actions):
        next_states, errors = self.model.predict(states, actions)
        return next_states, errors

    def predict_nextstates(self, states, actions):
        next_states = self.model.predict_nextstates(states, actions)
        return next_states

    def predict_errors(self, states, actions, predictions):
        errors = self.model.predict_errors(states, actions, predictions)
        return errors

    def predict_reward(self, states, actions):
        rewards = self.reward_model.predict(states, actions)
        return rewards

    def predict_done(self, states, actions):
        dones = self.done_model.predict(states, actions)
        return dones

    def evaluate_beliefs(self, states, actions, next_states):
        return self.model.evaluate_beliefs(states, actions, next_states)

    def to_device(self, device):
        self.model.to_device(device)

    def write_tensorboard_reward_done(self, tensorboard):
        tensorboard.add_scalar('/reward/error', self.reward_err()[0], self.total_steps)
        tensorboard.add_scalar('/done/error', self.done_err()[0], self.total_steps)
