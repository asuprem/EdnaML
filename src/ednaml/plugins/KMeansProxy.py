from ednaml.plugins import ModelPlugin
import torch
from sklearn.neighbors import KDTree

class KMeansProxy(ModelPlugin):
    def __init__(self, num_clusters, dimensions, dist, rand_seed, epochs):
        super().__init__(num_clusters = num_clusters, dimensions=dimensions, dist=dist, rand_seed = rand_seed, epochs=epochs)


    def build_plugin(self, **kwargs):
        self.num_clusters = kwargs.get("num_clusters")
        self.rand_seed = kwargs.get("rand_seed", 145845)
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist", "l2")    # TODO
        self.epochs = kwargs.get("epochs", 2)
        self.epoch_count = 0
        self.activated = False
        self.kdcluster = None

    
    def build_params(self, **kwargs):
        self.cluster_means = torch.rand(self.num_clusters, self.dimensions)
        self.cluster_counts = torch.zeros(self.num_clusters)


    def post_forward(self, x, feature_logits, features, secondary_outputs, **kwargs):
        if not self.activated:
            # perform the training here
            self.update_minibatch_kmeans_clusters(features)
            
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            dist, labels = self.compute_labels(features)
            return feature_logits, features, secondary_outputs, kwargs, {"cluster_mean": labels, "distance": dist}

    def post_epoch(self, epoch: int = 0, **kwargs):
        if not self.activated:  # If already activated, we do not need to change anything
            self.epoch_count = epoch    # Maybe have a better check in case something drops or epochs get skipped...
            self.activated = self.epoch_count > self.epochs # better way to deal with this whole situation, i.e. once activated, never changes...!
        if self.activated and self.kdcluster is not None:
            self.kdcluster = KDTree(self.cluster_means)


    def update_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        V = torch.zeros(self.num_clusters)
        idxs = torch.empty(batch.shape[0], dtype=torch.int)
        for j, x in enumerate(batch):
            idxs[j] = torch.argmin(((self.cluster_means - x)**2).sum(1))

        # Update centers:
        for j, x in enumerate(batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            self.cluster_means[idxs[j]] = (1.0 - eta) * self.cluster_means[idxs[j]] + eta * x

    def compute_labels(self, batch):
        """Compute the cluster labels for dataset X given centers C.
        """
        # labels = np.argmin(pairwise_distances(C, X), axis=0) # THIS REQUIRES TOO MUCH MEMORY FOR LARGE X
        dist, labels = self.kdcluster.query(batch, k=1, return_distance=True)   #.squeeze()
        return dist, labels


"""cluster_distances = torch.zeros(self.num_clusters)
    for cluster in range(self.num_clusters):
        cluster_distances[cluster] = sum(torch.sqrt((features - self.cluster_means[cluster])**2))
    c = torch.argmin(cluster_distances)
    self.cluster_counts[c] += 1
    self.cluster_means[c] += 1.0/self.cluster_counts[c]*(features - self.cluster_means[c])

"""