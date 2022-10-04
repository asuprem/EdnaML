import ednaml.core.decorators as edna

from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins import ModelPlugin
import torch
import torch.nn.functional
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
import h5py, os
from ednaml.plugins.ModelPlugin import ModelPlugin

@edna.register_model_plugin
class FastKMP(ModelPlugin):
    name = "FastKMP"
    def __init__(self, proxies = 10, dimensions = 768, dist = "euclidean", rand_seed = 34623498, iterations = 25, batch_size = 256, alpha = 0.5, feature_file = None):
        super().__init__(proxies = proxies, dimensions = dimensions, dist = dist, rand_seed = rand_seed, iterations = iterations, batch_size = batch_size, alpha = alpha, feature_file = feature_file)


    def build_plugin(self, **kwargs):
        self.proxies = kwargs.get("proxies")
        self.rand_seed = kwargs.get("rand_seed")
        self.dimensions = kwargs.get("dimensions")
        self.dist = kwargs.get("dist")    # TODO
        self.batch_size = kwargs.get("batch_size")
        self.iterations = kwargs.get("iterations")
        self.alpha = kwargs.get("alpha")

        self.activated = False
        self._dist_func = self.build_dist(self.dist)
        if self.dist == "euclidean":
            self._preprocess = lambda x:x
        elif self.dist == "cosine":
            self._preprocess = torch.nn.functional.normalize

        output_file = kwargs.get("feature_file")
        if output_file is None:
            import uuid
            output_file = "-".join(["FKMP", "features", uuid.uuid1().urn.split("-")[0].split(":")[-1]])
        self._output_file = output_file + ".h5"
        if os.path.exists(self._output_file):
            os.remove(self._output_file)
        self._writer = h5py.File(self._output_file, "w")  #we will delete old training features file
        self._written = False
        self._prev_idx = -1


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
        self.inertia = 0
        self.cluster_counts = torch.zeros(self.proxies)
        self.kdcluster = None
        self.high_density_thresholds = []


    def post_forward(self, x, feature_logits, features, secondary_outputs, model, **kwargs):
        if not self.activated:
            # perform the training here
            self.save_features(features)
            
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            dist, labels, idx = self.compute_labels(features)
            return feature_logits, features, secondary_outputs, kwargs, {"threshold": labels, "distance": dist, "label": idx}

    def save_features(self, features):
        feats = features.cpu().numpy()
        if self._written:
            self._writer["features"].resize((self._writer["features"].shape[0] + feats.shape[0]), axis=0)
            self._writer["features"][-feats.shape[0]:] = feats        
        else:   # First time writing -- we will need to create the dataset.
            self._writer.create_dataset("features", data=feats, compression = "gzip", chunks=True, maxshape=(None,feats.shape[1]))
            self._written = True

    def post_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        try:    # in case it is closed already somehow...
            self._writer.close()
        except:
            pass
        if not self.activated:  # If already activated, we do not need to change anything
            self.post_epoch_num = epoch    # Maybe have a better check in case something drops or epochs get skipped...
            if self.pre_epoch_flag: # Means we had a pre_epoch flag so an epoch was completed.
                self.pre_epoch_flag = False # reset the flag
                if self.post_epoch_num != self.pre_epoch_num:
                    raise ValueError("Epoch may have been skipped. Before: %i\tAfter: %i"%(self.pre_epoch_num, self.post_epoch_num))
            
            self.performkmeans()
            self.highdensitybins(model)
            
            self.activated = True

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
            for idx in indices[:,0]:
                distance_bins[idx].append(dist[idx, 0])
        
        self.high_density_thresholds = [None]*self.proxies
        import numpy as np
        for proxy in range(self.proxies):
            self.high_density_thresholds[proxy] = np.percentile(distance_bins[proxy], self.alpha * 100)
        print("Completed High Density threshold estimation")
        data.close()

    def compute_labels(self, features):
        """Compute the cluster labels for dataset X given centers C.
        """
        # labels = np.argmin(pairwise_distances(C, X), axis=0) # THIS REQUIRES TOO MUCH MEMORY FOR LARGE X
        feats = features.cpu()
        dist, idx = self.kdcluster.query(self._preprocess(feats), k=1, return_distance=True)   #.squeeze()
        # TODO convert idx to the actual cluster means to the actual cluster labels...
        return torch.from_numpy(dist).squeeze(1), torch.tensor([self.high_density_thresholds[item[0]] for item in idx]), idx[:,0]


    def pre_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        self.pre_epoch_flag = True
        self.pre_epoch_num = epoch

from ednaml.deploy import BaseDeploy

@edna.register_deployment
class FNCPluginDeployment(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    prediction_scores, pooled_out, outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    
    return prediction_scores, pooled_out, outputs
  def output_setup(self, **kwargs):
    pass
  def output_step(self, logits, features, secondary):
    pass