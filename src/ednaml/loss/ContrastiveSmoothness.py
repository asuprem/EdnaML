from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from ednaml.loss import Loss


class ContrastiveSmoothness(Loss):
    """Contrastive smoothness loss

    Calculates the contrastive loss of a mini-batch.

    Methods:
        __call__: Returns loss given features and labels.
    """

    def __init__(self, **kwargs):
        super(ContrastiveSmoothness, self).__init__()
        self.nmargin = kwargs.get("negative-margin", 0.5)
        self.pmargin = kwargs.get("positive-margin", 0.3)

    def forward(self, features, labels, epoch):
        positive_pairs, negative_pairs = self.get_pairs(features, labels)
        s_positive_pairs = F.log_softmax(features[positive_pairs[:,0]]).cuda()
        k_positive_pairs = F.log_softmax(features[positive_pairs[:,1]]).cuda()
        
        return torch.max(F.kl_div(s_positive_pairs, k_positive_pairs) + F.kl_div(k_positive_pairs, s_positive_pairs))/labels.shape[1]

    def get_pairs(self, features, labels):

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[
            (labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()
        ]
        negative_pairs = all_pairs[
            (labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()
        ]

        return positive_pairs, negative_pairs
