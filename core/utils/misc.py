import datetime
import time
# import matplotlib.pyplot as plt

from .torch_utils import *


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    total_episodes = 0
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join(config.get_modeldir(), 'model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag)))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            total_episodes += len(rewards)
            config.tensorboard.add_scalar('/dqn/reward/average_reward', np.mean(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/median_reward', np.median(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/min_reward', np.min(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/max_reward', np.max(rewards), agent.total_steps)
            config.logger.info('total steps %d, total episodes %3d, returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s' % (
                agent.total_steps, total_episodes, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards), len(rewards),
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
            t0 = time.time()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break
        agent.step()


def run_laplace_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    total_episodes = 0
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save()
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            total_episodes += len(rewards)

            if agent.num_episodes > agent.config.cold_start:
                total_loss = np.mean(agent.total_loss)
                attractive_loss = np.mean(agent.attractive_loss)
                repulsive_loss = np.mean(agent.repulsive_loss)
                agent.total_loss = []
                agent.attractive_loss = []
                agent.repulsive_loss = []
                config.tensorboard.add_scalar('/laplace/loss/total_loss', total_loss, total_episodes)
                config.tensorboard.add_scalar('/laplace/loss/attractive_loss', attractive_loss, total_episodes)
                config.tensorboard.add_scalar('/laplace/loss/repulsive_loss', repulsive_loss, total_episodes)
                config.logger.info('total steps %d, total episodes %3d, loss %.10f/%.10f/%.10f (total/attractive/repuslive), %.2f steps/s' % (
                    agent.total_steps, total_episodes, total_loss, attractive_loss, repulsive_loss,
                    config.log_interval / (time.time() - t0)))
                t0 = time.time()
        if agent.num_episodes > agent.config.cold_start and config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
            t0 = time.time()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break
        agent.step()


def run_steps_plan(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    total_episodes = 0
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join(config.get_modeldir(), 'model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag)))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            total_episodes += len(rewards)

            config.tensorboard.add_scalar('/dqn/reward/average_reward', np.mean(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/median_reward', np.median(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/min_reward', np.min(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/max_reward', np.max(rewards), agent.total_steps)

            rr_oracle, rr = 1.0, 1.0
            if agent.total_steps > config.exploration_steps:

                ratio_reject_oracle = agent.planner.get_statistics_oracle()

                rr_oracle = ratio_reject_oracle[-1]
                for k, t in enumerate(agent.planner.thresholds):
                    config.tensorboard.add_scalar('/plan/ratio_reject/threshold_{}_oracle'.format(t), ratio_reject_oracle[k],
                                                  agent.total_steps)

                if config.planner != 'PlannerOracle':
                    ratio_reject = agent.planner.get_statistics()
                    rr = ratio_reject[-1]
                    for k, t in enumerate(agent.planner.thresholds):
                        config.tensorboard.add_scalar('/plan/ratio_reject/threshold_{}'.format(t), ratio_reject[k], agent.total_steps)


                # config.tensorboard.add_scalar('/model/done_accuracy'.format(t),
                #                               agent.planner.get_done_acc(), agent.total_steps)
                # config.tensorboard.add_scalar('/model/reward_accuracy'.format(t),
                #                               agent.planner.get_reward_acc(), agent.total_steps)
            config.logger.info('total steps %d, total episodes %3d, returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), rr_oracle %.3f, rr %.3f, %.2f steps/s' % (
                agent.total_steps, total_episodes, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards), len(rewards),
                rr_oracle, rr,
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
            t0 = time.time()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break

        if config.save_error_dist and (agent.total_steps + 1) % config.error_dist_save_interval == 0:
            agent.planner.save_errors(config.get_error_dist_dir(), agent.total_steps + 1)

        agent.step()


def run_multi_steps_plan(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    total_episodes = 0
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(os.path.join(config.get_modeldir(), 'model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag)))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            total_episodes += len(rewards)

            config.tensorboard.add_scalar('/dqn/reward/average_reward', np.mean(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/median_reward', np.median(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/min_reward', np.min(rewards), agent.total_steps)
            config.tensorboard.add_scalar('/dqn/reward/max_reward', np.max(rewards), agent.total_steps)

            # if agent.total_steps > config.exploration_steps:
            #
            #     # ratio_reject_oracle = agent.planner.get_statistics_oracle()
                # for k, t in enumerate(agent.planner.thresholds):
                #     config.tensorboard.add_scalar('/plan/ratio_reject/threshold_{}_oracle'.format(t), ratio_reject_oracle[k],
                #                                   agent.total_steps)
                #
                # if config.planner != 'PlannerOracle':
                #     ratio_reject = agent.planner.get_statistics()
                #     for k, t in enumerate(agent.planner.thresholds):
                #         config.tensorboard.add_scalar('/plan/ratio_reject/threshold_{}'.format(t), ratio_reject[k], agent.total_steps)
                #

                # config.tensorboard.add_scalar('/model/done_accuracy'.format(t),
                #                               agent.planner.get_done_acc(), agent.total_steps)
                # config.tensorboard.add_scalar('/model/reward_accuracy'.format(t),
                #                               agent.planner.get_reward_acc(), agent.total_steps)
            config.logger.info('total steps %d, total episodes %3d, returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s' % (
                agent.total_steps, total_episodes, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards), len(rewards),
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
            t0 = time.time()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break

        if config.save_error_dist and (agent.total_steps + 1) % config.error_dist_save_interval == 0:
            agent.planner.save_errors(config.get_error_dist_dir(), agent.total_steps + 1)

        agent.step()


def determ_run_steps_no_disc(model, train_step=0):
    t0 = time.time()
    model.model.num_batches = len(model.cfg.data_loader)
    for ep in range(int(10e6)):
        for i, batch in enumerate(model.cfg.data_loader):
            model.train_step_determ_model(batch)
            if model.cfg.train_reward_done:
                model.train_reward_done_model(batch)
            if model.cfg.store_model and train_step % model.cfg.save_interval == 0 and train_step >= 0:
                model.store(training_step=model.total_steps)

            if not model.total_steps % model.cfg.log_interval:
                mdl = model.model.write_tensorboard_determ(model.cfg.tensorboard)
                log_str = "total steps %d, MSE-loss: %.8f, %.2f steps/s"
                model.cfg.logger.info(log_str % (model.total_steps, mdl, model.cfg.log_interval / (time.time() - t0)))
                t0 = time.time()
                if model.cfg.train_reward_done:
                    model.write_tensorboard_reward_done(model.cfg.tensorboard)

            if train_step == model.cfg.training_steps:
                quit()
            train_step += 1


def disc_run_steps(model):
    t0 = time.time()
    train_step = 0
    model.model.num_batches = len(model.cfg.data_loader)
    print (model.model.num_batches)
    for ep in range(int(10e6)):
        for i, batch in enumerate(model.cfg.data_loader):
            if model.cfg.store_model and train_step % model.cfg.save_interval == 0 and train_step >= 0:
                model.store_disc(training_step=train_step)
            model.train_step_disc_only(batch)
            if not model.total_steps % model.cfg.log_interval:
                log_str = (
                    "total steps %d, mse-loss: %.6f, error-loss: %.12f, %.2f steps/s"
                )
                me, mdl = model.model.write_tensorboard(model.cfg.tensorboard)
                model.cfg.logger.info(log_str % (
                    model.total_steps, mdl, me,
                    model.cfg.log_interval / (time.time() - t0)))
                t0 = time.time()

            if train_step == model.cfg.training_steps:
                quit()
            train_step += 1


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def parse_dataset(training_data):
    states = np.stack(training_data[:, 0]).reshape(-1, 10, 10, 4)
    actions = np.stack(training_data[:, 1])
    rewards = np.stack(training_data[:, 2])
    next_states = np.stack(training_data[:, 3]).reshape(-1, 10, 10, 4)
    dones = np.stack(training_data[:, 4]).astype(bool)
    return states, actions, rewards, next_states, dones


def convert_tensors(states, actions, rewards, next_states, dones):
    states = torch.from_numpy(states[~dones]).permute(0, 3, 1, 2).float()
    actions = torch.from_numpy(actions[~dones])
    rewards = torch.from_numpy(rewards[~dones])
    next_states = torch.from_numpy(next_states[~dones]).permute(0, 3, 1, 2).float()
    return states, actions, rewards, next_states
