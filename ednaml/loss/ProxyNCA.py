import torch
from torch import nn
from ednaml.loss import Loss
from ednaml.utils.math import pairwise_distance


class ProxyNCA(Loss):
    """ProxyNCA Loss

    Performs softmax with label smoothing.

    Args (kwargs only):
        classes (int): Number of classes in training set
        embedding_size (int): Number of embedding features
        smoothing (float, 0.3): Smoothing constant. Default 0.0
        normalization (float, 3.0): Normalization constant. Default 0.0

    Methods: 
        __call__: Returns loss given features and labels.

    """

    def __init__(self, **kwargs):
        super(ProxyNCA, self).__init__()

        self.classes = kwargs.get("classes")
        self.embedding = kwargs.get("embedding_size")
        self.DIV_CONST = 8
        self.SMOOTHING = kwargs.get("smoothing", 0.1)
        self.NORMALIZATION = kwargs.get("normalization", 3.0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.proxies = nn.Parameter(
            torch.randn(self.classes, self.embedding) / 8
        ).cuda()

    def forward(self, features, labels):
        normalized_proxy = self.NORMALIZATION * torch.nn.functional.normalize(
            self.proxies, p=2, dim=-1
        )
        normalized_logits = self.NORMALIZATION * torch.nn.functional.normalize(
            features, p=2, dim=-1
        )

        dist = pairwise_distance(
            torch.cat([normalized_logits, normalized_proxy]), squared=True
        )
        # the pairwise_distance is for all logits and proxies to each other. We just need logits to proxies, so we take the bottom left quadrant.
        dist = dist[: features.size()[0], features.size()[0] :]
        log_probs = self.logsoftmax(dist)

        # smooth labels
        labels = torch.zeros(log_probs.size()).scatter_(
            1, labels.unsqueeze(1).data.cpu(), 1
        )
        labels = labels.cuda()
        labels = (
            1 - (self.SMOOTHING + (self.SMOOTHING / self.embedding))
        ) * labels + self.SMOOTHING / self.embedding

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = (-labels * log_probs).mean(0).sum()
        return loss
