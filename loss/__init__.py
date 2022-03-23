from torch import nn
class Loss(nn.Module):
    def __init__(self, lossname=None, metadata=None, **kwargs):
            super().__init__()
    def forward(self):
        raise NotImplementedError()

from .builders import CoLabelLossBuilder, ClassificationLossBuilder, ReIDLossBuilder, CarZamLossBuilder