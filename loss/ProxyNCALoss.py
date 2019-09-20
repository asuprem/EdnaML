import torch
from torch import nn
from . import Loss
from utils.math import pairwise_distance

class ProxyNCA:
    def __init__(self,**kwargs):
        self.classes = kwargs.get("classes")
        self.embedding = kwargs.get("embedding_size")
        self.DIV_CONST = 8
        self.SMOOTHING = kwargs.get("smoothing", 0.0)
        self.NORMALIZATION = kwargs.get("normalization", 3.0)
        self.proxy = nn.Parameter(torch.randn(self.classes, self.embedding)/self.DIV_CONST)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def __call__(self,logits, labels):
        normalized_proxy = self.NORMALIZATION * torch.nn.functional.normalize(self.proxy,p=2,dim=-1)
        normalized_logits = self.NORMALIZATION * torch.nn.functional.normalize(logits, p = 2, dim = -1)
        dist = pairwise_distance(torch.cat([normalized_logits, normalized_proxy]), squared=True)
        # the pairwise_distance is for all logits and proxies to each other. We just need logits to proxies, so we take the bottom left quadrant.
        dist = dist[:logits.size()[0], logits.size()[0]:]
        log_probs = self.logsoftmax(dist)

        # smooth labels
        labels = torch.zeros(log_probs.size()).scatter_(1, labels.unsqueeze(1).data.cpu(), 1)
        labels = labels.cuda()
        labels = (1 - (self.eps + (self.eps / self.soft_dim))) * labels + self.eps / self.soft_dim

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = (- labels * log_probs).mean(0).sum()
        return loss
