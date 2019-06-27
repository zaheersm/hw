import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(DeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=5, padding=1)
        self.fcpre = nn.Linear(800, 1024)
        self.fca = nn.Linear(num_actions, 1024)
        self.fcpost = nn.Linear(1024, 800)
        self.tconv1 = nn.ConvTranspose2d(50, 50, kernel_size=5, padding=1)
        self.tconv2 = nn.ConvTranspose2d(50, 50, kernel_size=5, padding=1)
        self.tconv3 = nn.ConvTranspose2d(50, in_channels, kernel_size=5, padding=1)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 800)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 50, 4, 4)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = self.tconv3(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class DiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(DiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.fcpre = nn.Linear(5000, 1024)
        self.fca = nn.Linear(num_actions, 1024)
        self.fcpost = nn.Linear(1024, 800)
        self.tconv1 = nn.Linear(800, 400)
        self.tconv2 = nn.Linear(400, 200)
        self.tconv3 = nn.Linear(200, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 5000)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = self.tconv3(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class SigmoidNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=3, channels=30):
        super(SigmoidNet, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels * 10 * 10, num_actions)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, a):
        x = self.bn1(F.relu(self.conv1(s)))
        x = self.bn2(F.relu(self.conv2(x) + x))
        x = x.view(-1, self.channels * 10 * 10)
        x = self.sigmoid(self.fc(x))
        return x[torch.arange(x.shape[0]), a.long()]

    def predict(self, S, A):
        return self(S)[torch.arange(S.shape[0]), A.long()]


class SmallDiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallDiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3)
        self.fcpre = nn.Linear(1024, 256)
        self.fca = nn.Linear(num_actions, 256)
        self.fcpost = nn.Linear(256, 128)
        self.tconv1 = nn.Linear(128, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 1024)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class SmallDeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallDeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.fcpre = nn.Linear(1600, 512)
        self.fca = nn.Linear(num_actions, 512)
        self.fcpost = nn.Linear(512, 1600)
        self.tconv1 = nn.ConvTranspose2d(16, 16, kernel_size=5, padding=2)
        self.tconv2 = nn.ConvTranspose2d(16, in_channels, kernel_size=5, padding=2)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 1600)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 16, 10, 10)
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class SmallishDiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallishDiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3)
        self.fcpre = nn.Linear(512, 256)
        self.fca = nn.Linear(num_actions, 256)
        self.fcpost = nn.Linear(256, 128)
        self.tconv1 = nn.Linear(128, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 512)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class SmallishDeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallishDeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=2)
        self.fcpre = nn.Linear(800, 256)
        self.fca = nn.Linear(num_actions, 256)
        self.fcpost = nn.Linear(256, 800)
        self.tconv1 = nn.ConvTranspose2d(8, 8, kernel_size=5, padding=2)
        self.tconv2 = nn.ConvTranspose2d(8, in_channels, kernel_size=5, padding=2)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 800)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 8, 10, 10)
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class Smallishv2DiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(Smallishv2DiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5)
        self.fcpre = nn.Linear(288, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 64)
        self.tconv1 = nn.Linear(64, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 288)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class Smallishv2DeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(Smallishv2DeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5)
        self.fcpre = nn.Linear(288, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 288)
        self.tconv1 = nn.ConvTranspose2d(8, 8, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(8, in_channels, kernel_size=5, padding=2)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 288)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 8, 6, 6)
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class Smallishv3DiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(Smallishv3DiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=2)
        self.fcpre = nn.Linear(800, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 64)
        self.tconv1 = nn.Linear(64, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 800)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class Smallishv3DeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(Smallishv3DeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=2)
        self.fcpre = nn.Linear(800, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 800)
        self.tconv1 = nn.ConvTranspose2d(8, 8, kernel_size=5, padding=2)
        self.tconv2 = nn.ConvTranspose2d(8, in_channels, kernel_size=5, padding=2)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 800)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 8, 10, 10)
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)



















class SmallerDeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallerDeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)
        self.fcpre = nn.Linear(144, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 144)
        self.tconv1 = nn.ConvTranspose2d(4, 4, kernel_size=5, padding=2)
        self.tconv2 = nn.ConvTranspose2d(4, in_channels, kernel_size=5)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x)).view(-1, 4, 6, 6)
        x = F.relu(self.tconv1(x))
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class SmallerDiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(SmallerDiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)
        self.fcpre = nn.Linear(144, 128)
        self.fca = nn.Linear(num_actions, 128)
        self.fcpost = nn.Linear(128, 144)

        self.fc = nn.Linear(144, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.fc(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)


class TinyDeterministicNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(TinyDeterministicNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)

        self.fcpre = nn.Linear(144, 16)
        self.fca = nn.Linear(num_actions, 16)
        self.fcpost = nn.Linear(16, 144)

        self.fc = nn.Linear(144, 144)

        self.tconv2 = nn.ConvTranspose2d(4, in_channels, kernel_size=5)

    def forward(self, x, c):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)

        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))

        x = F.relu(self.fc(x)).view(-1, 4, 6, 6)
        x = self.tconv2(x)

        return x

    def predict(self, states, actions):
        return self(states, actions)

    def generate(self, states, actions):
        return self(states, actions)


class TinyDiscNet(nn.Module):
    """
        This is the best architecture I've found so far

    """
    def __init__(self, in_channels=4, num_actions=6):
        super(TinyDiscNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5)
        self.fcpre = nn.Linear(144, 16)
        self.fca = nn.Linear(num_actions, 16)
        self.fcpost = nn.Linear(16, 64)

        self.fc = nn.Linear(64, 64)
        self.tconv2 = nn.Linear(64, 1)

    def forward(self, x, c):

        x = F.relu(self.conv1(x))
        x = x.view(-1, 144)
        x = self.fcpre(x)
        a = (self.fca(c))
        x = x * a
        x = (self.fcpost(x))
        x = F.relu(self.fc(x))
        x = self.tconv2(x)

        return x

    def discriminate(self, S, one_hot_A, S_):
        return self(torch.cat((S, S_), dim=-3), one_hot_A)
