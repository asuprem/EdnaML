from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from ednaml.loss import Loss


class ContrastiveLoss(Loss):
    """Standard contrastive loss

    Calculates the contrastive loss of a mini-batch.

    Args (kwargs only):
        margin (float, 0.3): Margin constraint to use in triplet loss. If not provided,
        mine (str): Mining method. Default 'hard'. Supports ['hard', 'all'].

    Methods:
        __call__: Returns loss given features and labels.
    """

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = kwargs.get("margin", 0.3)

    def forward(self, features, labels):
        positive_pairs, negative_pairs = self.get_pairs(features, labels)
        if features.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (
            (features[positive_pairs[:, 0]] - features[positive_pairs[:, 1]])
            .pow(2)
            .sum(1)
        )
        negative_loss = F.relu(
            self.margin
            - (features[negative_pairs[:, 0]] - features[negative_pairs[:, 1]])
            .pow(2)
            .sum(1)
            .sqrt()
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

    def get_pairs(self, features, labels):
        distance_matrix = self.pdist(features)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[
            (labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()
        ]
        negative_pairs = all_pairs[
            (labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()
        ]

        negative_distances = distance_matrix[
            negative_pairs[:, 0], negative_pairs[:, 1]
        ]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(
            negative_distances, len(positive_pairs)
        )[: len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

    def pdist(self, vectors):
        distance_matrix = (
            -2 * vectors.mm(torch.t(vectors))
            + vectors.pow(2).sum(dim=1).view(1, -1)
            + vectors.pow(2).sum(dim=1).view(-1, 1)
        )
        return distance_matrix
