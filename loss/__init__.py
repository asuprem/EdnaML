from torch import nn
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self):
        raise NotImplementedError()

from .builders import ReIDLossBuilder
from .builders import CarZamLossBuilder

"""
Loss groups

Metric
    Triplet - hard, all
    Margin - 
    ProxyNCA
    ProxyTriplet (Not implemented)
    CompactContrastiveLos
    ContrastiveLoss

Compaction Loss
    ClusterLoss
    CenterLoss .0005


Classification Loss
    SoftmaxLogitsLoss - 
    SoftmaxLabelSmooth - 
"""