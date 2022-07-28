from ednaml.plugins import ModelPlugin
import torch
from sklearn.neighbors import KDTree

class KMeansProxy(ModelPlugin):
    name = "KMeansProxy"
    def __init__(self, num_clusters=10, dimensions=768, dist="l2", rand_seed=12344, epochs=3):
        super().__init__(num_clusters = num_clusters, dimensions=dimensions, dist=dist, rand_seed = rand_seed, epochs=epochs)


    def build_plugin(self, **kwargs):
        self.num_clusters = kwargs.get("num_clusters")
        self.rand_seed = kwargs.get("rand_seed", 145845)
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist", "l2")    # TODO
        self.epochs = kwargs.get("epochs", 2)
        self.epoch_count = 0
        self.pre_epoch_flag = False
        self.pre_epoch_num = -1
        self.post_epoch_num = -1
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
            self.post_epoch_num = epoch    # Maybe have a better check in case something drops or epochs get skipped...
            if self.pre_epoch_flag: # Means we had a pre_epoch flag so an epoch was completed.
                self.pre_epoch_flag = False # reset the flag
                self.epoch_count+=1
                if self.post_epoch_num != self.pre_epoch_num:
                    raise ValueError("Epoch may have been skipped. Before: %i\tAfter: %i"%(self.pre_epoch_num, self.post_epoch_num))
            self.activated = self.epoch_count > self.epochs # better way to deal with this whole situation, i.e. once activated, never changes...!
        if self.activated and self.kdcluster is None:
            self.kdcluster = KDTree(self.cluster_means)

    def pre_epoch(self, epoch: int = 0, **kwargs):
        self.pre_epoch_flag = True
        self.pre_epoch_num = epoch

    def update_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        #batch_device = batch.get_device()   # TODO fix this hacky with the hack from https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently 2nd answer
        local_batch = batch.cpu()
        V = torch.zeros(self.num_clusters)
        idxs = torch.empty(batch.shape[0], dtype=torch.int)
        for j, x in enumerate(local_batch):
            idxs[j] = torch.argmin(((self.cluster_means - x)**2).sum(1))

        # Update centers:
        for j, x in enumerate(local_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            self.cluster_means[idxs[j]] = (1.0 - eta) * self.cluster_means[idxs[j]] + eta * x

    def compute_labels(self, batch):
        """Compute the cluster labels for dataset X given centers C.
        """
        # labels = np.argmin(pairwise_distances(C, X), axis=0) # THIS REQUIRES TOO MUCH MEMORY FOR LARGE X
        q_batch = batch.cpu()
        dist, labels = self.kdcluster.query(q_batch, k=1, return_distance=True)   #.squeeze()
        return dist, labels


"""cluster_distances = torch.zeros(self.num_clusters)
    for cluster in range(self.num_clusters):
        cluster_distances[cluster] = sum(torch.sqrt((features - self.cluster_means[cluster])**2))
    c = torch.argmin(cluster_distances)
    self.cluster_counts[c] += 1
    self.cluster_means[c] += 1.0/self.cluster_counts[c]*(features - self.cluster_means[c])

"""