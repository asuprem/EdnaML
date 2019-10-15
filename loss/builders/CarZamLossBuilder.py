from . import LossBuilder

from ..ProxyNCALoss import ProxyNCA
from ..CompactContrastiveLoss import CompactContrastiveLoss
from ..CenterLoss import CenterLoss

class CarZamLossBuilder(LossBuilder):
    LOSS_PARAMS={}
    LOSS_PARAMS['ProxyNCA'] = {}
    LOSS_PARAMS['ProxyNCA']['fn'] = ProxyNCA
    LOSS_PARAMS['ProxyNCA']['args'] = ['features', 'labels']
    LOSS_PARAMS['CompactContrastiveLoss'] = {}
    LOSS_PARAMS['CompactContrastiveLoss']['fn'] = CompactContrastiveLoss
    LOSS_PARAMS['CompactContrastiveLoss']['args'] = ['features', 'labels', 'epoch']
    LOSS_PARAMS['CenterLoss'] = {}
    LOSS_PARAMS['CenterLoss']['fn'] = CenterLoss
    LOSS_PARAMS['CenterLoss']['args'] = ['features', 'labels']
 

    