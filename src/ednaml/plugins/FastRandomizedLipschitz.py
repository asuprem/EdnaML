
from typing import List
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins import ModelPlugin
import torch
import torch.nn.functional
from sklearn.neighbors import KDTree
from ednaml.plugins.ModelPlugin import ModelPlugin
from sortedcontainers import SortedKeyList
import numpy as np, os, h5py
from sklearn.cluster import MiniBatchKMeans


class FastRandomizedLipschitz(ModelPlugin):
    name = "FastRandomizedLipschitz"
    def __init__(self, proxies=10, dimensions=768, dist="euclidean", 
                            rand_seed=12344, neighbors = 10, proxy_epochs=3, 
                            perturbation_neighbors = 10, iterations = 25, batch_size = 256, 
                            alpha = 0.5, feature_file = None, classifier_access = "classifier", **kwargs):
        super().__init__(proxies = proxies, dimensions=dimensions, dist=dist, 
                            rand_seed = rand_seed, neighbors = neighbors, proxy_epochs=proxy_epochs, 
                            perturbation_neighbors=perturbation_neighbors, iterations = iterations, batch_size = batch_size, 
                            alpha = alpha, feature_file = feature_file, classifier_access = classifier_access)


    def build_plugin(self, **kwargs):
        self.proxies = kwargs.get("proxies")
        self.rand_seed = kwargs.get("rand_seed", 12344)
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist", "euclidean")    # TODO
        self.proxy_epochs = kwargs.get("proxy_epochs", 3)
        self.neighbors = kwargs.get("neighbors", 10)
        self.perturbation_neighbors = kwargs.get("perturbation_neighbors", self.neighbors)

        self.batch_size = kwargs.get("batch_size")
        self.iterations = kwargs.get("iterations")
        self.alpha = kwargs.get("alpha")

        output_file = kwargs.get("feature_file")
        if output_file is None:
            import uuid
            output_file = "-".join(["FRL", "features", uuid.uuid1().urn.split("-")[0].split(":")[-1]])
        self._output_file = output_file + ".h5"
        if os.path.exists(self._output_file):
            os.remove(self._output_file)
        self._writer = h5py.File(self._output_file, "w")  #we will delete old training features file
        self._written = False
        self._prev_idx = -1

        self.epoch_count = 0
        self.pre_epoch_flag = False
        self.pre_epoch_num = -1
        self.post_epoch_num = -1
        self.activated = False
        self.kdcluster = None
        self.proxy_stage = True
        self.lipschitz_stage = False

        self.classifier_access = kwargs.get("classifier_access")
        self._classifier_setup = False
        self._classifier = None

        
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
        self.inertia = 0
        self.cluster_counts = torch.zeros(self.proxies)
        self.kdcluster = None
        self.high_density_thresholds = []

        self.lipschitz_threshold = None
        self.lipschitz_threshold_mean = None
        self.smooth_lipschitz_threshold = None
        self.smooth_lipschitz_threshold_mean = None
        self.epsilon = None
        self.proxy_label = []


    def post_forward(self, x, feature_logits, features, secondary_outputs, model, **kwargs):
        if not self._classifier_setup:
            self._classifier = {mname:mlayer for mname, mlayer in model.named_modules()}[self.classifier_access]
            self._classifier_setup = True
        if not self.activated:
            # perform the training here
            if self.proxy_stage:
                self.save_features(features)
            if self.lipschitz_stage:
                # this epoch, we are trying to find the m closest points to each cluster mean...
                # for the batch, find each points distance to cluster means...??????
                self.search_minibatch_kmeans_clusters(features)
                
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            lscore, smoothlscore, proxy_labels = self.compute_lscores(features, feature_logits, model)
            # here, we use the features, perturb them, compute the logits, then compute l_score, then compare, etc, etc
            #dist, labels = self.compute_labels(features)
            return feature_logits, features, secondary_outputs, kwargs, {"l_score": lscore, "smooth_l_score": smoothlscore, 
                                                                            "l_threshold": [self.lipschitz_threshold]*features.shape[0], 
                                                                            "smooth_l_threshold": [self.smooth_lipschitz_threshold]*features.shape[0],
                                                                            "proxy_label": proxy_labels}


    def compute_lscores(self, features, feature_logits, model):
        perturbed = self.generate_perturbation(self.epsilon,self.dimensions,self.perturbation_neighbors).T.cuda()
        lscore = [None]*features.shape[0]
        smooth_lscore = [None]*features.shape[0]
        for j,x in enumerate(features):
            perturbation = x+perturbed
            with torch.no_grad():
                perturbed_logits = self._classifier(perturbation)
                raw_logits = feature_logits[j].unsqueeze(0)

                feature_lscore = torch.sqrt((perturbed**2).sum(1))
                logit_lscore = torch.sqrt(((perturbed_logits - raw_logits)**2).sum(1))

                smoothlogit_lscore = perturbed_logits[:,torch.argmax(raw_logits)] - raw_logits[:,torch.argmax(raw_logits)]

                l_scores = logit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]
                smooth_lscores = smoothlogit_lscore[feature_lscore > 0] / feature_lscore[feature_lscore > 0]

                lscore[j] = torch.max(l_scores).cpu().item()
                smooth_lscore[j] = torch.max(smooth_lscores).cpu().item()
        
        _, idx = self.kdcluster.query(self._preprocess(features.cpu()), k=1, return_distance=True)   #.squeeze()
        
        return lscore, smooth_lscore, self.proxy_label[idx[:,0]]


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

                    try:    # in case it is closed already somehow...
                        self._writer.close()
                    except:
                        pass
                    
                    self.performkmeans()
                    self.highdensitybins(model)

                    with torch.no_grad():
                        self.proxy_label = torch.argmax(self._classifier(self.cluster_means.cuda()).cpu(), dim=1)
                    self.high_density_thresholds  = torch.tensor(self.high_density_thresholds)
                    
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
                        raw_logits = self._classifier(self.cluster_means[idx].unsqueeze(0).cuda()).cpu()
                        lipschitz_logits = self._classifier(torch.stack([item[0] for item in self._closest_features[idx]]).cuda()).cpu()

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

    def save_features(self, features):
        feats = features.cpu().numpy()
        if self._written:
            self._writer["features"].resize((self._writer["features"].shape[0] + feats.shape[0]), axis=0)
            self._writer["features"][-feats.shape[0]:] = feats        
        else:   # First time writing -- we will need to create the dataset.
            self._writer.create_dataset("features", data=feats, compression = "gzip", chunks=True, maxshape=(None,feats.shape[1]))
            self._written = True

    def performkmeans(self):
        import time
        inertia = []
        data = h5py.File(self._output_file, 'r')
        data_size = data['features'].shape[0]
        
        print("Starting KMeans for k={kval}, with {iters} iterations".format(kval=self.proxies, iters=self.iterations))
        kmeans = MiniBatchKMeans(n_clusters = self.proxies, random_state = self.rand_seed, batch_size = self.batch_size)
        stime = time.time()
        for iters in range(self.iterations):
            for i in range(0, data_size, self.batch_size):
                current_data = data['features'][i:i+self.batch_size]
                kmeans.partial_fit(current_data)
            if iters%5 == 0:
                etime = round(time.time() - stime, 2)
                print("\t[{elapse} s] -- Completed {iters} iterations".format(iters=iters, elapse = etime))
                stime = time.time()
        print("\tCompeted MBKM for k={kval}, with inertia: {inertia}".format(kval=self.proxies, inertia = kmeans.inertia_))
        data.close()
        self.cluster_means = torch.tensor(kmeans.cluster_centers_)
        self.inertia = kmeans.inertia_
        return inertia

    def highdensitybins(self, model: ModelAbstract):
        if self.dist == "euclidean":
            self.kdcluster = KDTree(self.cluster_means)
        elif self.dist == "cosine":
            self.kdcluster = KDTree(torch.nn.functional.normalize(self.cluster_means))
        else:
            raise ValueError("Invalid value for dist: %s"%self.dist)

        data = h5py.File(self._output_file, 'r')
        data_size = data['features'].shape[0]
        distance_bins = [[] for _ in range(self.proxies)]
        print("Starting High Density estimation")
        for i in range(0, data_size, self.batch_size):
            current_data = data['features'][i:i+self.batch_size]
            dist, indices = self.kdcluster.query(self._preprocess(torch.tensor(current_data)), k=1, return_distance=True)   #.squeeze()
            for idx,val in enumerate(indices[:,0]):
                distance_bins[val].append(dist[idx, 0])
        
        self.high_density_thresholds = [None]*self.proxies
        import numpy as np
        for proxy in range(self.proxies):
            self.high_density_thresholds[proxy] = np.percentile(distance_bins[proxy], self.alpha * 100)
        print("Completed High Density threshold estimation")
        data.close()


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