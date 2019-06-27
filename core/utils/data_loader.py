import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from core.utils.torch_utils import tensor


class ToTensor(object):
    def __init__(self, dims):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dims = dims

    def __call__(self, sample):
        x, y, h = self.dims
        return {'state': tensor(sample['state'].reshape(x, y, h), self.device).permute(2, 0, 1),
                'action': tensor(sample['action'], self.device).long(),
                'reward': tensor(sample['reward'], self.device).double(),
                'next_state': tensor(sample['next_state'].reshape(x, y, h), self.device).permute(2, 0, 1),
                'done': tensor(sample['done'], self.device).byte()}


class MinAtarTransitions(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = list(map(lambda fname: os.path.join(data_dir, fname),
                              os.listdir(self.data_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # print(file_path)
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
            sample = dict()
            sample['state'], sample['action'], sample['reward'], sample['next_state'], sample['done'] = instance
            if self.transform:
                sample = self.transform(sample)
            return sample


# if __name__ == '__main__':
#
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
#     g = MinAtarTransitions(os.path.join(project_root, 'data',
#                                                   'output', 'mini_atari', 'breakout', 'data_10k'),
#                            transform=ToTensor())
#
#     data_loader = DataLoader(g, batch_size=128, shuffle=True, num_workers=1)
#     print("Dataset size: {}".format(len(g)))
#     for i_batch, b in enumerate(data_loader):
#         # print(i_batch, sampled_batch['state'], sampled_batch['action'], sampled_batch['done'])
#         # sampled_batch['state'][~sampled_batch['done']]
#         break