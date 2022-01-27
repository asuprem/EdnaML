from loss import Loss
from . import LossBuilder

from ..SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ..SoftmaxLabelSmooth import SoftmaxLabelSmooth



class CoLabelLossBuilder(LossBuilder):
    LOSS_PARAMS = {}
    LOSS_PARAMS['SoftmaxLogitsLoss'] = {}
    LOSS_PARAMS['SoftmaxLogitsLoss']['fn'] = SoftmaxLogitsLoss
    LOSS_PARAMS['SoftmaxLogitsLoss']['args'] = ['logits', 'labels']
    LOSS_PARAMS['SoftmaxLabelSmooth'] = {}
    LOSS_PARAMS['SoftmaxLabelSmooth']['fn'] = SoftmaxLabelSmooth
    LOSS_PARAMS['SoftmaxLabelSmooth']['args'] = ['logits', 'labels']
