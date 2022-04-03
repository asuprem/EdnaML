import torch
from ednaml.loss import Loss


class SoftmaxLogitsLoss(Loss):
    """Standard softmax loss

    Calculates the softmax loss.

    Methods: 
        __call__: Returns loss given logits and labels.

    """

    def forward(self, logits, labels):
        """
    Args:
        logits: prediction matrix (before softmax) with shape (batch_size, softmax_dimensions)
        labels: ground truth labels with shape (batch_size)
    """
        return torch.nn.functional.cross_entropy(
            logits[labels >= 0], labels[labels >= 0]
        )
