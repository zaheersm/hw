from ..utils import *


class Agent:
    def __init__(self, config):
        self.config = config

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = env.step([action])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.config.timeout:
                break
        return total_rewards

    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))

    def eval_step(self, state):
        raise Exception('eval_step not implemented')

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)



