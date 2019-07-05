from datetime import datetime

from .normalizer import *
from .misc import *


class SlimConfig:
    def __init__(self):
        self.run = 0
        self.param_setting = 0
        self.exp_name = 'test'
        self.timeout = None
        self.task_fn = None
        self.logger = None
        self.tag = 'vanilla'
        self.max_steps = 0
        self.__eval_env = None
        self.state_dim = None
        self.action_dim = None
        self.task_name = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 3
        self.data_root = None

        self.network_type = 'flat'
        self.use_target_network = True


        self.laplace_representation = False
        self.modulation = False
    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

    # Helper functions
    def get_logdir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        ensure_dir(d)
        return d

    def get_error_dist_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "error_dist")
        ensure_dir(d)
        return d


    # Helper functions
    def get_data_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "data_500k_expert_nosticky")
        ensure_dir(d)
        return d

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))

    def get_tflogdir(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting),
                            "tf_logs", datetime.now().strftime('%D-%T'))

    # Helper functions
    def get_modeldir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "model")
        ensure_dir(d)
        return d

    def log_config(self, logger):
        # serializes the configuration into a log file
        attrs = self.get_print_attrs()
        for param, value in sorted(attrs.items(), key=lambda x: x[0]):
            logger.info('{}: {}'.format(param, value))

    def get_heatmapdir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting), "heatmaps")
        ensure_dir(d)
        return d


class DQNAgentConfig(SlimConfig):
    def __init__(self):
        super(DQNAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = None
        self.state_norm_coef = 1.0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0


    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class LaplaceAgentConfig(SlimConfig):
    def __init__(self):
        # TODO: Clean this up. It has relics of DQNAgentConfig which are not used in LaplaceAgent
        super(LaplaceAgentConfig, self).__init__()
        self.agent = 'LaplaceAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = None
        self.state_norm_coef = 1.0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

        self.vector_type = 'flat'

        self.multi_head = False


    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs

    # Helper functions
    def get_eigvecdir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "eigvecdir")
        ensure_dir(d)
        return d


class DQNLaplaceAgentConfig(SlimConfig):
    def __init__(self):
        super(DQNLaplaceAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = None
        self.state_norm_coef = 1.0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

        self.laplace_representation = True
        self.vector_type = 'flat'
        self.multi_head = False

    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class LaplaceRepresentationConfig(SlimConfig):
    def __init__(self):
        super(LaplaceRepresentationConfig, self).__init__()
        self.agent = 'LaplaceRepresentationAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.q_batch_size = None
        self.q_opt_type = 'RMSProp'
        self.q_opt_fn = None
        self.q_replay_fn = None
        self.q_memory_size = None
        self.q_net_type = 'flat'
        self.q_net_fn = None

        self.l_batch_size = None
        self.l_opt_type = 'RMSProp'
        self.l_opt_fn = None
        self.l_replay_fn = None
        self.l_memory_size = None
        self.vector_type = 'flat'
        self.vector_fn = None
        self.laplace_representation = True
        self.multi_head = False

        self.q_net_fn = None
        self.target_network_update_freq = None

        self.state_normalizer = None
        self.state_norm_coef = 1.0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'q_opt_fn', 'q_net_fn', 'q_replay_fn', 'data_root',
                  'l_opt_fn', 'vector_fn', 'l_replay_fn']:
            del attrs[k]
        return attrs