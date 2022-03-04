import pdb
from torch import nn
from .abstracts import CoLabelInterpretableResnetAbstract
from utils import layers
import torch

# CHANGELOG: secondary attention is List, not a single number

class CoLabelInterpretableResnet(CoLabelInterpretableResnetAbstract):
    """Basic CoLabel Resnet model.

    A CoLabel model is a base ResNet, but during prediction, employs additional pieces such as 
    an ensemble voter, a heuristic based on the prediction output probabilities, as well as (if desired), 
    holistic nested side inputs.

    Args: (TODO)
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.
        soft_dimensions (int, None): Whether to include softmax classification layer. If None, softmax layer is not created in model.

    Kwargs (MODEL_KWARGS):
        last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
        attention (str, None): The attention module to use. Only supports ['cbam', None]
        input_attention (bool, false): Whether to include the IA module
        secondary_attention (List[int], None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic blocks provided here in List.
        branches (int): How many complementary feature branches for this inteprretable model

    Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS):
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

    def __init__(self, base = 'interpretable_resnet50', weights=None, normalization=None, embedding_dimensions=None, soft_dimensions = None, **kwargs):
        super(CoLabelInterpretableResnet, self).__init__(base, weights, normalization, embedding_dimensions, soft_dimensions=soft_dimensions, **kwargs)

    def build_base(self,base, weights, **kwargs):
        """Build the model base.

        Builds the architecture base/core.
        """
        _resnet = __import__("backbones.interpretableresnet", fromlist=["interpretableresnet"])
        _resnet = getattr(_resnet, base)
        # Set up the resnet backbone
        self.base = _resnet(last_stride=1, **kwargs)
        if weights is not None:
            self.base.load_param(weights)
        
        # TODO add branches here...
        for branch_idx in self.branches:
            self.branch_layers[branch_idx]["gap"] = nn.AdaptiveAvgPool2d(1)
            #self.branch_layers[branch_idx]["fc"] = nn.Linear(512 * self.base.block.expansion, self.branch_classes[branch_idx])
                
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.emb_linear = torch.nn.Identity()
        # Refactor this later, but basically, we don't need a linear embedding layer to convert from featurs to features. We will go directly from features to softmax...
        if self.embedding_dimensions is None:
            self.embedding_dimensions = 512*self.base.block.expansion
        if self.embedding_dimensions > 512 * self.base.block.expansion:
            raise Warning("You are trying to scale up embedding dimensions from %i to %i. Try using same or less dimensions."%(512*self.base.block.expansion, self.embedding_dimensions))
        elif self.embedding_dimensions == 512*self.base.block.expansion:
            pass
        else:
            raise Warning("You are trying to scale down embedding dimensions from %i to %i. Try using same or less dimensions."%(512*self.base.block.expansion, self.embedding_dimensions))
            #self.emb_linear = nn.Linear(self.base.block.expansion*512, self.embedding_dimensions, bias=False)

    def build_normalization(self, normalization):
        norm_func=nn.Module
        norm_args={}
        norm_div=1
        if self.normalization == 'bn':
            norm_func = nn.BatchNorm1d
            norm_args={"affine":True}
        elif self.normalization == "in":
            norm_func = layers.FixedInstanceNorm1d
            norm_args={"affine":True}
        elif self.normalization == "gn":
            norm_div=16
            norm_func = nn.GroupNorm
            norm_args={"num_channels":self.embedding_dimensions, "affine":True}
        elif self.normalization == "ln":
            norm_func = nn.LayerNorm
            norm_args={"elementwise_affine":True}
        elif self.normalization == 'l2':
            norm_func = layers.L2Norm
            norm_args={"scale":1.0}            
        elif self.normalization is None or self.normalization == '':
            norm_func = layers.L2Norm
            norm_args={"scale":1.0}
        else:
            raise NotImplementedError()

        # Not implemented should have been raised by now, so we don't need to worry about it here...
        self.feat_norm = norm_func(self.embedding_dimensions // norm_div, **norm_args)
        for branch_idx in self.branches:
            self.branch_layers[branch_idx]["feat_norm"] = norm_func(self.embedding_dimensions//norm_div, **norm_args)


        if self.normalization == 'l2' or self.normalization == '' or self.normalization is None:
            pass
        elif self.normalization == 'bn' or self.normalization == 'gn' or self.normalization == 'ln' or self.normalization == 'in':
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
            for branch_idx in self.branches:
                self.branch_layers[branch_idx]["feat_norm"].bias.requires_grad_(False)
                self.branch_layers[branch_idx]["feat_norm"].apply(self.weights_init_kaiming)
        else:
            raise NotImplementedError()

    # here, we are updating build_softmax so that we can controlt the different branch outputs as well...
    def build_softmax(self):
        for branch_idx in self.branches:
            self.branch_layers[branch_idx]["softmax"] = nn.Linear(512 * self.base.block.expansion, self.branch_classes[branch_idx])
            self.branch_layers[branch_idx]["softmax"].apply(self.weights_init_softmax)

        if self.soft_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.soft_dimensions, bias=False)
            self.softmax.apply(self.weights_init_softmax)
        else:
            raise Warning("soft_dimensions is None. This should be fixed, otherwise there will be no predictions from model")
            self.softmax = None

    def base_forward(self,x):
        
        concat_features, branch_features = self.base(x)
        
        concat_features = self.gap(concat_features)
        concat_features = concat_features.view(concat_features.shape[0],-1)
        concat_features = self.emb_linear(concat_features)  #identify, for now NOTE

        for branch_idx in self.branches:
            branch_features[branch_idx] = self.branch_layers[branch_idx]["gap"](branch_features[branch_idx])
            branch_features[branch_idx] = branch_features[branch_idx].view(branch_features[branch_idx].shape[0],-1)
            #branch_features[branch_idx] = nn.Identity()...

        return concat_features, branch_features

    def forward(self,x):
        concat_features, branch_features = self.base_forward(x)
        
        #if self.feat_norm is not None: <-- no need, identity
        concat_inference = self.feat_norm(concat_features)
        for branch_idx in self.branches:
            branch_features[branch_idx] = self.branch_layers[branch_idx]["feat_norm"](branch_features[branch_idx])

        soft_logits = None
        branch_logits = [None]*self.branches
        
        soft_logits = self.softmax(concat_inference)
        for branch_idx in self.branches:
            branch_logits[branch_idx] = self.branch_layers[branch_idx]["softmax"](branch_features[branch_idx])
        #if self.softmax:
        #    soft_logits = self.softmax(inference)
        #return soft_logits, inference   # soft logits are the softmax logits we will use to for training. We can use inference to store the historical probability????
        return soft_logits, branch_logits