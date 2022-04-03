from torch import nn


class Loss(nn.Module):
    def __init__(self, lossname=None, metadata=None, **kwargs):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


from ednaml.loss.CenterLoss import CenterLoss
from ednaml.loss.ClusterLoss import ClusterLoss
from ednaml.loss.CompactContrastiveLoss import CompactContrastiveLoss
from ednaml.loss.ContrastiveLoss import ContrastiveLoss
from ednaml.loss.MarginLoss import MarginLoss
from ednaml.loss.ProxyNCA import ProxyNCA
from ednaml.loss.SoftmaxLabelSmooth import SoftmaxLabelSmooth
from ednaml.loss.SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ednaml.loss.TripletLoss import TripletLoss
