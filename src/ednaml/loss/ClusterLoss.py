import torch
import torch.nn.functional as F
from ednaml.loss import Loss


class ClusterLoss(Loss):
    def __init__(self, margin: float, instances: int, images_per_instance: int):
        """ ClusterLoss from Cluster Loss for Person Re-Identification

        Args:
            margin (float): margin for cluster loss
            instances (int): Number of unique ids in a batch
            images_per_instance (int): Number of images per instance in batch. Should be batch_size / instances
        """
        super(ClusterLoss, self).__init__()
        self.margin = margin
        self.ids_per_batch = instances
        self.imgs_per_id = images_per_instance

    def _euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _cluster_loss(self, features, targets, ids_per_batch=32, imgs_per_id=4):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             cluster_loss
        """

        unique_labels = targets.cpu().unique().cuda()

        inter_min_distance = torch.zeros(unique_labels.size(0))
        intra_max_distance = torch.zeros(unique_labels.size(0))
        center_features = torch.zeros(unique_labels.size(0), features.size(1))

        inter_min_distance = inter_min_distance.cuda()
        intra_max_distance = intra_max_distance.cuda()
        center_features = center_features.cuda()

        index = torch.range(0, unique_labels.size(0) - 1)
        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
            intra_class_distance = self._euclidean_dist(
                center_features[index == i], same_class_features
            )
            # print('intra_class_distance', intra_class_distance)
            intra_max_distance[i] = intra_class_distance.max()
        # print('intra_max_distance:', intra_max_distance)

        for i in range(unique_labels.size(0)):
            inter_class_distance = self._euclidean_dist(
                center_features[index == i], center_features[index != i]
            )
            # print('inter_class_distance', inter_class_distance)
            inter_min_distance[i] = inter_class_distance.min()
        #  print('inter_min_distance:', inter_min_distance)
        cluster_loss = torch.mean(
            torch.relu(intra_max_distance - inter_min_distance + self.margin)
        )
        return cluster_loss, intra_max_distance, inter_min_distance

    def forward(self, features, labels):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            labels: ground truth labels with shape (batch_size)

        Return:
             cluster_loss
        """
        cluster_loss, cluster_dist_ap, cluster_dist_an = self._cluster_loss(
            features, labels, self.ids_per_batch, self.imgs_per_id
        )
        return cluster_loss
