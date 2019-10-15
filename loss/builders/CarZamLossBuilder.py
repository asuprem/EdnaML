from . import LossBuilder

from ..ProxyNCALoss import ProxyNCA
from ..CompactContrastiveLoss import CompactContrastiveLoss


class CarZamLossBuilder(LossBuilder):
    LOSS_PARAMS={}
    LOSS_PARAMS['ProxyNCA'] = {}
    LOSS_PARAMS['ProxyNCA']['fn'] = ProxyNCA
    LOSS_PARAMS['ProxyNCA']['args'] = ['features', 'proxies', 'labels']
    LOSS_PARAMS['CompactContrastiveLoss'] = {}
    LOSS_PARAMS['CompactContrastiveLoss']['fn'] = CompactContrastiveLoss
    LOSS_PARAMS['CompactContrastiveLoss']['args'] = ['features', 'labels', 'epoch']
 

    