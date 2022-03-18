from torch import nn
class Loss(nn.Module):
    def __init__(self, **kwargs):
            super(Loss, self).__init__()
    def forward(self):
        raise NotImplementedError()

from .builders import CoLabelLossBuilder, ClassificationLossBuilder, ReIDLossBuilder, CarZamLossBuilder