
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins import ModelPlugin
import torch
import torch.nn.functional
from sklearn.neighbors import KDTree
from ednaml.plugins.ModelPlugin import ModelPlugin


class RandomizedLipschitz(ModelPlugin):
    name = "RandomizedLipschitz"
    def __init__(self, proxies=10, dimensions=768, dist="euclidean", rand_seed=12344, neighbors = 10, proxy_epochs=3):
        super().__init__(proxies = proxies, dimensions=dimensions, dist=dist, rand_seed = rand_seed, neighbors = 10, proxy_epochs=proxy_epochs)


    def build_plugin(self, **kwargs):
        self.proxies = kwargs.get("proxies")
        self.rand_seed = kwargs.get("rand_seed", 12344)
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist", "euclidean")    # TODO
        self.proxy_epochs = kwargs.get("proxy_epochs", 3)
        self.neighbors = kwargs.get("neighbors", 10)

        self.epoch_count = 0
        self.pre_epoch_flag = False
        self.pre_epoch_num = -1
        self.post_epoch_num = -1
        self.activated = False
        self.kdcluster = None
        self.proxy_stage = True
        self.lipschitz_stage = False
        
        self._dist_func = self.build_dist(self.dist)
        if self.dist == "euclidean":
            self._preprocess = lambda x:x
        elif self.dist == "cosine":
            self._preprocess = torch.nn.functional.normalize



    def build_dist(self, dist="euclidean"):
        if dist == "euclidean":
            return self.l2dist
        elif dist == "cosine":
            return self.cosdist
        else:
            raise ValueError("Invalid value for dist: %s"%dist)

    def l2dist(self, x):
        return torch.argmin(torch.sqrt(((self.cluster_means - x)**2).sum(1)))
    def cosdist(self, x):   #https://stackoverflow.com/questions/46409846/using-k-means-with-cosine-similarity-python
        return self.l2dist(torch.nn.functional.normalize(x, dim=0))
        
    
    def build_params(self, **kwargs):
        self.cluster_means = torch.rand(self.proxies, self.dimensions)
        self.cluster_counts = torch.zeros(self.proxies)


    def post_forward(self, x, feature_logits, features, secondary_outputs, **kwargs):
        if not self.activated:
            # perform the training here
            if self.proxy_stage:
                self.update_minibatch_kmeans_clusters(features)
            if self.lipschitz_stage:
                # this epoch, we are trying to find the m closest points to each cluster mean...
                # for the batch, find each points distance to cluster means...??????
                import pdb
                pdb.set_trace()
                self.search_minibatch_kmeans_clusters(features)
                
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            dist, labels = self.compute_labels(features)
            return feature_logits, features, secondary_outputs, kwargs, {"cluster_mean": labels, "distance": dist}


    def post_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        if not self.activated:  # If already activated, we do not need to change anything
            
            if self.proxy_stage:    # Check if we are ready to move to lipschitz stage.
                self.post_epoch_num = epoch    # Maybe have a better check in case something drops or epochs get skipped...
                if self.pre_epoch_flag: # Means we had a pre_epoch flag so an epoch was completed.
                    self.pre_epoch_flag = False # reset the flag
                    self.epoch_count+=1
                    if self.post_epoch_num != self.pre_epoch_num:
                        raise ValueError("Epoch may have been skipped. Before: %i\tAfter: %i"%(self.pre_epoch_num, self.post_epoch_num))
                self.lipschitz_stage = self.epoch_count > self.proxy_epochs # better way to deal with this whole situation, i.e. once activated, never changes...!
                self.proxy_stage = not self.lipschitz_stage
            elif self.lipschitz_stage:
                # check if we are done and can activate 
                pass
        if (self.activated or self.lipschitz_stage) and self.kdcluster is None:
            if self.dist == "euclidean":
                self.kdcluster = KDTree(self.cluster_means)
            elif self.dist == "cosine":
                self.kdcluster = KDTree(torch.nn.functional.normalize(self.cluster_means))
            else:
                raise ValueError("Invalid value for dist: %s"%self.dist)

            #model.classifier(self.cluster_means.cuda())
            #self.labels = torch.argmax(model.classifier(self.cluster_means.cuda()), dim=1).cpu()   # potentially unneeded


    def pre_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        self.pre_epoch_flag = True
        self.pre_epoch_num = epoch

    def update_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        #batch_device = batch.get_device()   # TODO fix this hacky with the hack from https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently 2nd answer
        local_batch = batch.cpu()
        V = torch.zeros(self.proxies)
        idxs = torch.empty(batch.shape[0], dtype=torch.int)
        for j, x in enumerate(local_batch):
            idxs[j] = self._dist_func(x)

        # Update centers:
        for j, x in enumerate(local_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            self.cluster_means[idxs[j]] = (1.0 - eta) * self.cluster_means[idxs[j]] + eta * x

    def search_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        #batch_device = batch.get_device()   # TODO fix this hacky with the hack from https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently 2nd answer
        local_batch = batch.cpu()
        V = torch.zeros(self.proxies)
        idxs = torch.empty(batch.shape[0], dtype=torch.int)
        for j, x in enumerate(local_batch):
            idxs[j] = self._dist_func(x)

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
        dist, idx = self.kdcluster.query(self._preprocess(q_batch), k=1, return_distance=True)   #.squeeze()
        # TODO convert idx to the actual cluster means to the actual cluster labels...
        return torch.from_numpy(dist).squeeze(1), torch.stack([self.labels[item[0]] for item in idx])