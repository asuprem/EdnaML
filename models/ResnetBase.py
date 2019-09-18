import pdb
import importlib
from torch import nn
from .abstracts import ReidModel
from utils import layers

class ResnetBase(ReidModel):
    def __init__(self, base = 'resnet50', weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        super(ResnetBase, self).__init__(base, weights, normalization, embedding_dimensions, soft_dimensions, **kwargs)
    
    def build_base(self,base, weights, **kwargs):
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
        elif self.normalization is None:
            self.feat_norm = None
        else:
            raise NotImplementedError()

    def base_forward(self,x):
        features = self.gap(self.base(x))
        features = features.view(features.shape[0],-1)
        if self.emb_linear is not None:
            features = self.emb_linear(features)
        return features

