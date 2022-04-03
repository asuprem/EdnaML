from torch import nn
import torch
from ednaml.loss import Loss


class SoftmaxLabelSmooth(Loss):
    """Softmax with label smoothing

    Performs softmax with label smoothing.

    Args (kwargs only):
        softmax_dimensions (int): Dimension of the softmax layer. Used for the smoothing constant scaling.
        eps (float): Smothing constant. Default 0.1

    Methods: 
        __call__: Returns loss given logits and labels.

    """

    def __init__(self, lossname, metadata, **kwargs):
        super().__init__()
        self.softmax_dimensions = kwargs.get("softmax_dimensions", None)
        if self.softmax_dimensions is None:
            self.softmax_dimensions = metadata.getLabelDimensions(lossname)
        self.eps = kwargs.get("eps", 0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, labels):
        """
        Args:
            logits: prediction matrix (before softmax) with shape (batch_size, softmax_dimensions)
            labels: ground truth labels with shape (batch_size)
        """
        llogits = logits[labels >= 0]
        llabels = labels[labels >= 0]
        log_probs = self.logsoftmax(llogits)
        llabels = torch.zeros(log_probs.size()).scatter_(
            1, llabels.unsqueeze(1).data.cpu(), 1
        )
        llabels = llabels.cuda()
        llabels = (1 - self.eps) * llabels + self.eps / self.softmax_dimensions
        loss = (-llabels * log_probs).mean(0).sum()
        return loss
