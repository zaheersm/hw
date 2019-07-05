import argparse

from tensorboardX import SummaryWriter

from core import *
from experiment.sweeper import Sweeper


def set_optimizer_fn(cfg):
    if cfg.q_opt_type == 'SGD':
        cfg.q_opt_fn = lambda params: torch.optim.SGD(params, cfg.q_lr)
    elif cfg.q_opt_type == 'RMSProp':
        cfg.q_opt_fn = lambda params: torch.optim.RMSprop(params, cfg.q_lr)
    else:
        raise NotImplementedError

    if cfg.l_opt_type == 'SGD':
        cfg.l_opt_fn = lambda params: torch.optim.SGD(params, cfg.l_lr)
    elif cfg.l_opt_type == 'RMSProp':
        cfg.l_opt_fn = lambda params: torch.optim.RMSprop(params, cfg.l_lr)
    elif cfg.l_opt_type == 'Adam':
        cfg.l_opt_fn = lambda params: torch.optim.Adam(params, cfg.l_lr)
    else:
        raise NotImplementedError


def set_qnet_fn(cfg):
    if cfg.q_net_type == 'flat' and cfg.laplace_representation:
        cfg.q_net_fn = lambda: VanillaNet(cfg.action_dim, FCBody(np.prod(cfg.d),
                                            hidden_units=tuple(cfg.hidden_units)), cfg.device)
    elif cfg.q_net_type == 'linear' and cfg.laplace_representation:
        cfg.q_net_fn = lambda: LinearNet(cfg.action_dim, cfg.d, cfg.device)

    else:
        raise NotImplementedError


def set_vector_fn(cfg):
    if cfg.vector_type == 'flat' and cfg.multi_head:
        cfg.vector_fn = lambda: VanillaNet(cfg.d, FCBody(np.prod(cfg.state_dim),
                                           hidden_units=tuple(cfg.vec_hidden_units)), cfg.device)
    elif cfg.vector_type == 'flat' and not cfg.multi_head:
        cfg.vector_fn = lambda: VanillaNet(1, FCBody(np.prod(cfg.state_dim),
                                           hidden_units=tuple(cfg.vec_hidden_units)), cfg.device)
    else:
        raise NotImplementedError


def set_task_fn(cfg):
    if 'mini_atari' in cfg.exp_name:
        return lambda: MiniAtariTask(cfg.task_name, args.id)
    elif cfg.task_name == 'OneRoom':
        return lambda: OneRoom(cfg.task_name, args.id)
    elif cfg.task_name == 'TwoRooms':
        return lambda: TwoRooms(cfg.task_name, args.id)
    elif cfg.task_name == 'HardMaze':
        return lambda: HardMaze(cfg.task_name, args.id)
    elif cfg.task_name == 'OneRoomLaplace':
        return lambda: OneRoomLaplace(cfg.task_name, args.id)
    elif cfg.task_name == 'TwoRoomsLaplace':
        return lambda: TwoRoomsLaplace(cfg.task_name, args.id)
    elif cfg.task_name == 'HardMazeLaplace':
        return lambda: HardMazeLaplace(cfg.task_name, args.id)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file', default='experiment/config_files/mini_atari/breakout/dqn/basic.json')
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--device', default=-1, type=int, )
    args = parser.parse_args()
    project_root = os.path.abspath(os.path.dirname(__file__))
    sweeper = Sweeper(os.path.join(project_root, args.config_file))

    cfg = sweeper.parse(args.id)
    cfg.data_root = os.path.join(project_root, 'data', 'output')
    set_one_thread()
    random_seed(args.id)
    cfg.device = select_device(args.device)

    # Setting up the config
    cfg.task_fn = set_task_fn(cfg)
    cfg.eval_env = cfg.task_fn()
    cfg.state_normalizer = RescaleNormalizerv2(cfg.state_norm_coef)
    cfg.reward_normalizer = RescaleNormalizer(cfg.reward_norm_coef)

    # Setting up the optimizer
    set_optimizer_fn(cfg)

    cfg.q_replay_fn = lambda: Replay(memory_size=int(cfg.q_memory_size), batch_size=cfg.q_batch_size)
    cfg.l_replay_fn = lambda: Replay(memory_size=int(cfg.l_memory_size), batch_size=cfg.l_batch_size)

    set_qnet_fn(cfg)
    set_vector_fn(cfg)

    cfg.random_action_prob = LinearSchedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_schedule_steps)

    # Setting up the logger
    logdir = cfg.get_logdir()
    log_path = os.path.join(logdir, 'log')
    cfg.logger = setup_logger(log_path, stdout=True)
    cfg.log_config(cfg.logger)

    writer = SummaryWriter(cfg.get_logdir())
    cfg.tensorboard = writer

    # Initializing the agent and running the experiment
    agent_class = getattr(agent, cfg.agent)
    agent = agent_class(cfg)
    if cfg.load_representation: agent.load(os.path.join(cfg.data_root, cfg.load_path))
    run_steps(agent)
