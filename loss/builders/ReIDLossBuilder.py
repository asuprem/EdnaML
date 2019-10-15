from . import LossBuilder

from ..SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ..SoftmaxLabelSmooth import SoftmaxLabelSmooth
from ..TripletLoss import TripletLoss
from ..MarginLoss import MarginLoss
from ..ContrastiveLoss import ContrastiveLoss
from ..CompactContrastiveLoss import CompactContrastiveLoss
from ..ProxyNCALoss import ProxyNCA
from ..CenterLoss import CenterLoss
from ..ClusterLoss import ClusterLoss

class ReIDLossBuilder(LossBuilder):
  LOSS_PARAMS = {}
  LOSS_PARAMS['SoftmaxLogitsLoss'] = {}
  LOSS_PARAMS['SoftmaxLogitsLoss']['fn'] = SoftmaxLogitsLoss
  LOSS_PARAMS['SoftmaxLogitsLoss']['args'] = ['logits', 'labels']
  LOSS_PARAMS['TripletLoss'] = {}
  LOSS_PARAMS['TripletLoss']['fn'] = TripletLoss
  LOSS_PARAMS['TripletLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['MarginLoss'] = {}
  LOSS_PARAMS['MarginLoss']['fn'] = MarginLoss
  LOSS_PARAMS['MarginLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['SoftmaxLabelSmooth'] = {}
  LOSS_PARAMS['SoftmaxLabelSmooth']['fn'] = SoftmaxLabelSmooth
  LOSS_PARAMS['SoftmaxLabelSmooth']['args'] = ['logits', 'labels']
  LOSS_PARAMS['ContrastiveLoss'] = {}
  LOSS_PARAMS['ContrastiveLoss']['fn'] = ContrastiveLoss
  LOSS_PARAMS['ContrastiveLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['CompactContrastiveLoss'] = {}
  LOSS_PARAMS['CompactContrastiveLoss']['fn'] = CompactContrastiveLoss
  LOSS_PARAMS['CompactContrastiveLoss']['args'] = ['features', 'labels', 'epoch']
  LOSS_PARAMS['ProxyNCA'] = {}
  LOSS_PARAMS['ProxyNCA']['fn'] = ProxyNCA
  LOSS_PARAMS['ProxyNCA']['args'] = ['features', 'labels']
  LOSS_PARAMS['CenterLoss'] = {}
  LOSS_PARAMS['CenterLoss']['fn'] = CenterLoss
  LOSS_PARAMS['CenterLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['ClusterLoss'] = {}
  LOSS_PARAMS['ClusterLoss']['fn'] = ClusterLoss
  LOSS_PARAMS['ClusterLoss']['args'] = ['features', 'labels']

