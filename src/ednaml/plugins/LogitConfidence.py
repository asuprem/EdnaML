import ednaml.core.decorators as edna
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins import ModelPlugin
import torch
import torch.nn.functional
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
import h5py, os
from ednaml.plugins.ModelPlugin import ModelPlugin

class LogitConfidence(ModelPlugin):
    name = "LogitConfidence"
    def __init__(self, num_classes = 2, **kwargs):
        super().__init__(num_classes = num_classes)

    def build_plugin(self, **kwargs):
        self.epochs = 1
        self.activated = False
        self.num_classes = kwargs.get("num_classes")

    def build_params(self, **kwargs):
        self.logit_confidence_logits = torch.zeros((1,self.num_classes))
        self.logit_confidence_count = torch.zeros((1,self.num_classes))
        self.logit_confidence = torch.zeros((1,self.num_classes))


    def post_forward(self, x, feature_logits, features, secondary_outputs, model, **kwargs):
        if not self.activated:
            # perform the training here
            self.add_to_average(feature_logits)
            
            return feature_logits, features, secondary_outputs, kwargs, {}
        else:
            logit, threshold = self.compute_labels(feature_logits)
            return feature_logits, features, secondary_outputs, kwargs, {"logit": logit, "logit_threshold": threshold}

    def add_to_average(self, feature_logits):
        # Basically, for each class, find the entries where they are the max, average those
        logits = feature_logits.cpu().numpy()
        max_idx = torch.argmax(logits, dim=1, keepdim=True)
        one_hot = torch.zeros(logits.shape)
        one_hot.scatter_(0, max_idx, 1)
        self.logit_confidence_logits += torch.sum(logits * one_hot, dim=0)
        self.logit_confidence_count += torch.sum(one_hot, dim=0)

    def post_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        if not self.activated:  # If already activated, we do not need to change anything
            # No need to do anything...
            self.logit_confidence = self.logit_confidence_logits / self.logit_confidence_count
            self.activated = True
        self.pre_epoch_flag = False

    
    def compute_labels(self, feature_logits):
        """Compute the cluster labels for dataset X given centers C.
        """
        # labels = np.argmin(pairwise_distances(C, X), axis=0) # THIS REQUIRES TOO MUCH MEMORY FOR LARGE X
        feats = feature_logits.cpu()
        return torch.max(feats, dim=1), self.logit_confidence[torch.argmax(feats, dim=1)]

    def pre_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        self.pre_epoch_flag = True
        self.pre_epoch_num = epoch

