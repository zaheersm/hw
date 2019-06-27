import numpy as np


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.maxes = None
        self.mins = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.maxes is None:
            self.maxes = np.zeros_like(x)
            self.mins = np.zeros_like(x)
        self.maxes = np.maximum(x, self.maxes)
        self.mins = np.minimum(x, self.mins)
        range = self.maxes - self.mins
        return np.true_divide(x, range, out=np.zeros_like(x), where=range != 0, casting='unsafe')


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        return self.coef * x

class RescaleNormalizerv2(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        return 2*self.coef*x - 1