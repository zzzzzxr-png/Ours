import torch.nn as nn


class Representation(nn.Module):
    def inverse(self, coefficients):
        raise NotImplementedError
