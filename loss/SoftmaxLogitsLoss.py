import torch
from . import Loss
class SoftmaxLogitsLoss(Loss):
  def __call__(self, logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)