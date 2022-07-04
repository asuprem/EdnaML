import importlib
from torch import nn
from .abstracts import ReidModel
from utils import layers

class ShuffleNetBase(ReidModel):
    """Basic ReID ShuffleNet model.

    A ReID model performs re-identification by generating embeddings such that the same class's embeddings are closer together.

    Args:
        base (str): The architecture base for shufflenet, i.e. shufflenetv2-small ONLY
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where no normalization is used. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output. If providing a value, it should be less than the architecture core's base feature dimensions.
        soft_dimensions (int, None): Whether to include softmax classification layer. If None, softmax layer is not created in model.
    
    Kwargs (MODEL_KWARGS):
        ia_attention (bool, false): Whether to include input IA module
        part_attention (bool, false): Whether to include Part-CBAM Mobule

TODO
    Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS; set in backbones.shufflenet):
        zero_init_residual (bool, false): Whether the final layer uses zero initialization
        top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        num_classes (int, 1000): Number of features in final imagenet FC layer
        groups (int, 1): Used during resnet variants construction
        width_per_group (int, 64): Used during resnet variants construction
        replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
        norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D

    Methods: 
        forward: Process a batch

    """
    def __init__(self, base = 'shufflenetv2_small', weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        super(ShuffleNetBase, self).__init__(base, weights, normalization, embedding_dimensions, soft_dimensions, **kwargs) #.abstracts.ReidModel
    
    def build_base(self,base, weights, **kwargs):
        _shufflenet = __import__("backbones.shufflenet", fromlist=["shufflenet"])
        _shufflenet = getattr(_shufflenet, base)    #base is shufflenetv2_small, e.g.
        self.base = _shufflenet(**kwargs)
        if weights is not None:
            self.base.load_param(weights)
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.emb_linear = None
        #1280 MAGIC - SHUFFLENET output
        if self.embedding_dimensions > 1280:
            raise ValueError("You are trying to scale up embedding dimensions from %i to %i. Try using same or less dimensions."%(1280, self.embedding_dimensions))
        elif self.embedding_dimensions == 1280:
            pass
        else:
            self.emb_linear = nn.Linear(1280, self.embedding_dimensions, bias=False)
    
    def build_normalization(self, normalization):
        """Build the mormalization layer.

        Args:
            normalization (str, None): Which normalization to use for the normalization layer between feature generation and feature inference.
                * None -- No normalization
                * 'bn' -- batch normalization
                * 'in' -- instancce normalization
                * 'gn' -- group normalization
                * 'ln' -- layer normalization
                * 'l2' -- l2 normalization

        """
        
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

    def forward(self,x):
        features = self.base_forward(x)
        
        if self.feat_norm is not None:
            inference = self.feat_norm(features)
        else:
            inference = features

        if self.training:
            if self.softmax is not None:
                soft_logits = self.softmax(inference)
            else:
                soft_logits = None
            return soft_logits, features
        else:
            return inference

