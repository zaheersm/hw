from ..utils import *

import pickle


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:

            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))

        return batch_data

    def sample_array(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]

        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def persist_memory(self, dir):
        for k in range(len(self.data)):
            transition = self.data[k]
            with open(os.path.join(dir, str(k)), "wb") as f:
                pickle.dump(transition, f)
        quit()


class TensorReplay:
    def __init__(self, memory_size, batch_size, device):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.step = 0
        self.device = device

        self.tensors = None
        self.buffer = []

    # def feed_batch(self, experience):
    #     batch_size = len(experience[0])
    #     assert self.memory_size % batch_size == 0
    #
    #     if self.tensors is None:
    #         self.tensors = experience
    #     else:
    #         if len(self.tensors[0]) < self.memory_size:
    #             for k in range(len(experience)):
    #                 self.tensors[k] = torch.cat([self.tensors[k], experience[k]])
    #         else:
    #             for k in range(len(experience)):
    #                 self.tensors[k][self.pos: self.pos+batch_size] = experience[k]
    #     self.pos = (self.pos + batch_size) % self.memory_size

    def feed_batch(self, experience):
        self.buffer.append(experience[0])
        if len(self.buffer) % 32 == 0:
            buffer = list(map(lambda x: np.asarray(x), zip(*self.buffer)))
            states, actions, rewards, next_states, dones = self.to_tensor(buffer)
            self.feed_tensors([states, actions, rewards, next_states, dones])
            self.buffer = []

    def feed_tensors(self, experience):
        batch_size = len(experience[0])
        assert self.memory_size % batch_size == 0
        if self.tensors is None:
            self.tensors = self.init_memory(experience)
        for k in range(len(experience)):
            self.tensors[k][self.pos: self.pos+batch_size] = experience[k]
        self.pos = (self.pos + batch_size) % self.memory_size
        self.step += batch_size

    def init_memory(self, experience):
        tensors = []
        for k in range(len(experience)):
            size = [self.memory_size] + list(experience[k].size()[1:])
            tensors.append(experience[k].new_empty(size=size, device=self.device))
        return tensors

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idx = torch.randint_like(torch.arange(batch_size),
                                 low=0, high=min(self.step, self.memory_size)).to(self.device)
        batch_data = []
        for k in range(len(self.tensors)):
            batch_data.append(self.tensors[k][idx])
        return batch_data

    def to_tensor(self, transitions):
        states, actions, rewards, next_states, dones = transitions
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).double().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).byte().to(self.device)
        return states, actions, rewards, next_states, dones

    def persist_memory(self, dir):
        s, a, r, ns, d = self.tensors
        for k in range(self.memory_size):
            transition = [s[k].numpy(), a[k].item(), r[k].item(), ns[k].numpy(), d[k].item()]
            with open(os.path.join(dir, str(k)), "wb") as f:
                pickle.dump(transition, f)
        quit()
