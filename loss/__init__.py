from torch import nn
class Loss(nn.Module):
    def __init__(self):
            super(Loss, self).__init__()
    def forward(self):
        raise NotImplementedError()

from .builders import ReIDLossBuilder
from .builders import CarZamLossBuilder