import pdb
import importlib
from torch import nn
from .abstracts import ReidModel
from utils import layers
import torch

class CarzamResnet(ReidModel):
    """Basic CarZam Resnet model.

    A CarZam model is similar to a Re-ID model. It yields a feature map of an input.

    Args:
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str): Path to weights file for the architecture base ONLY.
        normalization (str): Cann be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimension (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.

    Methods: 
        forward: Process a batch (TODO add type and shape information)

    """
    def __init__(self, base = 'resnet50', weights=None, normalization=None, embedding_dimensions=None, **kwargs):
        super(CarzamResnet, self).__init__(base, weights, normalization, embedding_dimensions, soft_dimensions=None, **kwargs)
    
    def build_base(self,base, weights, **kwargs):
        """Build the model base.

        Builds the architecture base/core.
        """
        _resnet = __import__("backbones.resnet", fromlist=["resnet"])
        _resnet = getattr(_resnet, base)
        self.base = _resnet(last_stride=1, **kwargs)
        if weights is not None:
            self.base.load_param(weights)
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.emb_linear = None
        if self.embedding_dimensions > 512 * self.base.block.expansion:
            raise ValueError("You are trying to scale up embedding dimensions from %i to %i. Try using same or less dimensions."%(512*self.base.block.expansion, self.embedding_dimensions))
        elif self.embedding_dimensions == 512*self.base.block.expansion:
            pass
        else:
            self.emb_linear = nn.Linear(self.base.block.expansion*512, self.embedding_dimensions, bias=False)
            # Initialization TODO
    
    def build_normalization(self, normalization):
        if self.normalization == 'bn':
            self.feat_norm = nn.BatchNorm1d(self.embedding_dimensions)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "in":
            self.feat_norm = layers.FixedInstanceNorm1d(self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "gn":
            self.feat_norm = nn.GroupNorm(self.embedding_dimensions // 16, self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "ln":
            self.feat_norm = nn.LayerNorm(self.embedding_dimensions,elementwise_affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == 'l2':
            self.feat_norm = layers.L2Norm(self.embedding_dimensions,scale=1.0)
        elif self.normalization is None or self.normalization == '':
            self.feat_norm = None
        else:
            raise NotImplementedError()


    def base_forward(self,x):
        features = self.gap(self.base(x))
        features = features.view(features.shape[0],-1)
        if self.emb_linear is not None:
            features = self.emb_linear(features)
        return features


    def forward(self,x):
        features = self.base_forward(x)
        
        if self.feat_norm is not None:
            inference = self.feat_norm(features)
        else:
            inference = torch.nn.functional.normalize(features, p = 2, dim = 1)
        return inference
        """
        if self.training:
            if self.softmax is not None:
                soft_logits = self.softmax(inference)
            else:
                soft_logits = None
            return soft_logits, features
        else:
            return inference
        """