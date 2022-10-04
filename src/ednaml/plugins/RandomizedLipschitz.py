
from typing import List
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins import ModelPlugin
import torch
import torch.nn.functional
from sklearn.neighbors import KDTree
from ednaml.plugins.ModelPlugin import ModelPlugin
from sortedcontainers import SortedKeyList
import numpy as np

class RandomizedLipschitz(ModelPlugin):
    name = "RandomizedLipschitz"
    def __init__(self, proxies=10, dimensions=768, dist="euclidean", rand_seed=12344, neighbors = 10, proxy_epochs=3, perturbation_neighbors = 10, **kwargs):
        super().__init__(proxies = proxies, dimensions=dimensions, dist=dist, rand_seed = rand_seed, neighbors = neighbors, proxy_epochs=proxy_epochs, perturbation_neighbors=perturbation_neighbors)


    def build_plugin(self, **kwargs):
        self.proxies = kwargs.get("proxies")
        self.rand_seed = kwargs.get("rand_seed", 12344)
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist", "euclidean")    # TODO
        self.proxy_epochs = kwargs.get("proxy_epochs", 3)
        self.neighbors = kwargs.get("neighbors", 10)
        self.perturbation_neighbors = kwargs.get("perturbation_neighbors", self.neighbors)

        self.epoch_count = 0
        self.pre_epoch_flag = False
        self.pre_epoch_num = -1
        self.post_epoch_num = -1
        self.activated = False
        self.kdcluster = None
        self.proxy_stage = True
        self.lipschitz_stage = False

        self.lipschitz_threshold = None
        self.lipschitz_threshold_mean = None
        self.smooth_lipschitz_threshold = None
        self.smooth_lipschitz_threshold_mean = None
        self.epsilon = None

        
        self._dist_func = self.build_dist(self.dist)
        if self.dist == "euclidean":
            self._preprocess = lambda x:x
        elif self.dist == "cosine":
            self._preprocess = torch.nn.functional.normalize

        self._closest_features : List[SortedKeyList] = None



    def build_dist(self, dist="euclidean"):
        if dist == "euclidean":
            return self.l2dist
        elif dist == "cosine":
            return self.cosdist
        else:
            raise ValueError("Invalid value for dist: %s"%dist)

    def l2dist(self, x):
        return torch.sqrt(((self.cluster_means - x)**2).sum(1)) # torch.argmin()
    def cosdist(self, x):   #https://stackoverflow.com/questions/46409846/using-k-means-with-cosine-similarity-python
        return self.l2dist(torch.nn.functional.normalize(x, dim=0))
        
    
    def build_params(self, **kwargs):
        self.cluster_means = torch.rand(self.proxies, self.dimensions)
        self.cluster_counts = torch.zeros(self.proxies)


    def post_forward(self, x, feature_logits, features, secondary_outputs, model, **kwargs):
        if not self.activated:
            # perform the training here
            if self.proxy_stage:
                self.update_minibatch_kmeans_clusters(features)
            if self.lipschitz_stage:
                # this epoch, we are trying to find the m closest points to each cluster mean...
                # for the batch, find each points distance to cluster means...??????
                self.search_minibatch_kmeans_clusters(features)
                
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            lscore, smoothlscore = self.compute_lscores(features, feature_logits, model)
            # here, we use the features, perturb them, compute the logits, then compute l_score, then compare, etc, etc
            #dist, labels = self.compute_labels(features)
            return feature_logits, features, secondary_outputs, kwargs, {"l_score": lscore, "smooth_l_score": smoothlscore, "l_threshold": [self.lipschitz_threshold]*features.shape[0], "smooth_l_threshold": [self.smooth_lipschitz_threshold]*features.shape[0]}


    def compute_lscores(self, features, feature_logits, model):
        perturbed = self.generate_perturbation(self.epsilon,self.dimensions,self.perturbation_neighbors).T.cuda()
        lscore = [None]*features.shape[0]
        smooth_lscore = [None]*features.shape[0]
        for j,x in enumerate(features):
            perturbation = x+perturbed
            with torch.no_grad():
                perturbed_logits = model.classifier(perturbation)
                raw_logits = feature_logits[j].unsqueeze(0)

                feature_lscore = torch.sqrt((perturbed**2).sum(1))
                logit_lscore = torch.sqrt(((perturbed_logits - raw_logits)**2).sum(1))

                smoothlogit_lscore = perturbed_logits[:,torch.argmax(raw_logits)] - raw_logits[:,torch.argmax(raw_logits)]

                l_scores = logit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]
                smooth_lscores = smoothlogit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]

                lscore[j] = torch.max(l_scores).cpu().item()
                smooth_lscore[j] = torch.max(smooth_lscores).cpu().item()
        
        return lscore, smooth_lscore


    def post_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        if not self.activated:  # If already activated, we do not need to change anything
            
            if self.proxy_stage:    # Check if we are ready to move to lipschitz stage.
                self.post_epoch_num = epoch    # Maybe have a better check in case something drops or epochs get skipped...
                if self.pre_epoch_flag: # Means we had a pre_epoch flag so an epoch was completed.
                    self.pre_epoch_flag = False # reset the flag
                    self.epoch_count+=1
                    if self.post_epoch_num != self.pre_epoch_num:
                        raise ValueError("Epoch may have been skipped. Before: %i\tAfter: %i"%(self.pre_epoch_num, self.post_epoch_num))
                self.lipschitz_stage = self.epoch_count >= self.proxy_epochs # better way to deal with this whole situation, i.e. once activated, never changes...!
                self.proxy_stage = not self.lipschitz_stage
                if self.proxy_stage:
                    self._logger.info("RandomizedLipschitz continuing proxy stage")
                else:
                    self._logger.info("RandomizedLipschitz starting Lipschitz stage")
            elif self.lipschitz_stage:
                # check if we are done and can activate 
                self._logger.info("RandomizedLipschitz has completed Lipschitz stage. Computing L values.")
                lthresh = [0]*len(self.cluster_means)
                lthreshmean = [0]*len(self.cluster_means)
                smooththresh = [0]*len(self.cluster_means)
                smooththreshmean = [0]*len(self.cluster_means)
                with torch.no_grad():
                    for idx in range(len(self.cluster_means)):
                        raw_logits = model.classifier(self.cluster_means[idx].unsqueeze(0).cuda()).cpu()
                        lipschitz_logits = model.classifier(torch.stack([item[0] for item in self._closest_features[idx]]).cuda()).cpu()

                        # use euclidean here, only!!!!!!
                        feature_lscore = torch.sqrt(((torch.stack([item[0] for item in self._closest_features[idx]])  - self.cluster_means[idx].unsqueeze(0))**2).sum(1))
                        logit_lscore = torch.sqrt(((lipschitz_logits - raw_logits)**2).sum(1))

                        smoothlogit_lscore = lipschitz_logits[:,torch.argmax(raw_logits)] - raw_logits[:,torch.argmax(raw_logits)]

                        l_scores = logit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]
                        smooth_lscores = smoothlogit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]
                        
                        lthresh[idx] = torch.max(l_scores).item()
                        lthreshmean[idx] = torch.mean(l_scores).item()
                        smooththresh[idx] = torch.max(smooth_lscores).item()
                        smooththreshmean[idx] = torch.mean(smooth_lscores).item()
                
                self.lipschitz_threshold = max(lthresh)
                self.lipschitz_threshold_mean = max(lthreshmean)
                self.smooth_lipschitz_threshold = max(smooththresh)
                self.smooth_lipschitz_threshold_mean = max(smooththreshmean)

                self.epsilon = 1. / self.smooth_lipschitz_threshold

                self.activated = True

                # So we are done settng up neighbors...
                # here, we have access to the model. We can not pass all our features and stuff through the model, get the logits, compute L, etc, etc
                

        if self.lipschitz_stage and self._closest_features is None and not self.activated:
            """
            if self.dist == "euclidean":
                self.kdcluster = KDTree(self.cluster_means)
            elif self.dist == "cosine":
                self.kdcluster = KDTree(torch.nn.functional.normalize(self.cluster_means))
            else:
                raise ValueError("Invalid value for dist: %s"%self.dist)
            """
            #model.classifier(self.cluster_means.cuda())
            #self.labels = torch.argmax(model.classifier(self.cluster_means.cuda()), dim=1).cpu()   # potentially unneeded
            # build the array of closest neighbor features...
            self._closest_features = [SortedKeyList(key=lambda x:x[1]) for _ in range(len(self.cluster_means))]  #x <- (feature, distance)


    def pre_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        self.pre_epoch_flag = True
        self.pre_epoch_num = epoch

    def update_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        #batch_device = batch.get_device()   # TODO fix this hacky with the hack from https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently 2nd answer
        local_batch = batch.cpu()
        V = torch.zeros(self.proxies)
        idxs = torch.empty(batch.shape[0], dtype=torch.int)
        for j, x in enumerate(local_batch):
            idxs[j] = torch.min(self._dist_func(x),0)[1].item()

        # Update centers:
        for j, x in enumerate(local_batch):
            V[idxs[j]] += 1
            eta = 1.0 / V[idxs[j]]
            self.cluster_means[idxs[j]] = (1.0 - eta) * self.cluster_means[idxs[j]] + eta * x

    def search_minibatch_kmeans_clusters(self, batch: torch.Tensor, ):
        #batch_device = batch.get_device()   # TODO fix this hacky with the hack from https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently 2nd answer
        local_batch = batch.cpu()
        
        for _, x in enumerate(local_batch):
            resp = self._dist_func(x)   # distance of x to each cluster_center
            # basically --> 
            [feature_list.add((x, resp[idx].item())) for idx, feature_list in enumerate(self._closest_features)]
        
        # prune each to be length of self.neighbors...
        self._closest_features = [SortedKeyList(feature_list[:self.neighbors], key=feature_list.key) for idx, feature_list in enumerate(self._closest_features)]
        
        # For each cluster center, we have a dict of closest features...

    def generate_perturbation(self, epsilon, n_dims, n_samples):
        Y = np.random.multivariate_normal(mean=[0], cov=np.eye(1,1), size=(n_dims, n_samples))
        Y = np.squeeze(Y, -1)
        Y /= np.sqrt(np.sum(Y * Y, axis=0))
        U = np.random.uniform(low=0, high=1, size=(n_samples)) ** (1/n_dims)
        Y *= U * epsilon # in my case radius is one
        return torch.Tensor(Y)