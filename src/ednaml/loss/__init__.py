from torch import nn

from ednaml.utils import locate_class


class Loss(nn.Module):
    def __init__(self, lossname=None, metadata=None, **kwargs):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class TorchLoss(Loss):
    def __init__(self, lossname=None, metadata=None, **kwargs):
        super().__init__()
        self.lossclass = kwargs.get("loss_class")
        self.losskwargs = kwargs.get("loss_kwargs")
        lossclass = locate_class(
            package="torch", subpackage="nn", classpackage=self.lossclass
        )
        self.lossfn = lossclass(**self.losskwargs)

    def forward(
        self, input, target, **kwargs
    ):  # TODO this is hacky. need to generalize, or find a better way to represent the torch loss with its arguments...
        return self.lossfn(input, target, **kwargs)


from ednaml.loss.CenterLoss import CenterLoss
from ednaml.loss.ClusterLoss import ClusterLoss
from ednaml.loss.CompactContrastiveLoss import CompactContrastiveLoss
from ednaml.loss.ContrastiveLoss import ContrastiveLoss
from ednaml.loss.MarginLoss import MarginLoss
from ednaml.loss.ProxyNCA import ProxyNCA
from ednaml.loss.SoftmaxLabelSmooth import SoftmaxLabelSmooth
from ednaml.loss.SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ednaml.loss.TripletLoss import TripletLoss
