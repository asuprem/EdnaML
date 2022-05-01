from ednaml.loss.builders import LossBuilder

from ednaml.loss import SoftmaxLogitsLoss, TorchLoss
from ednaml.loss import SoftmaxLabelSmooth


class ClassificationLossBuilder(LossBuilder):
    LOSS_PARAMS = {}
    LOSS_PARAMS["SoftmaxLogitsLoss"] = {}
    LOSS_PARAMS["SoftmaxLogitsLoss"]["fn"] = SoftmaxLogitsLoss
    LOSS_PARAMS["SoftmaxLogitsLoss"]["args"] = ["logits", "labels"]
    LOSS_PARAMS["SoftmaxLabelSmooth"] = {}
    LOSS_PARAMS["SoftmaxLabelSmooth"]["fn"] = SoftmaxLabelSmooth
    LOSS_PARAMS["SoftmaxLabelSmooth"]["args"] = ["logits", "labels"]
    LOSS_PARAMS["TorchLoss"] = {}
    LOSS_PARAMS["TorchLoss"]["fn"] = TorchLoss
    LOSS_PARAMS["TorchLoss"]["args"] = ["input", "target"]
