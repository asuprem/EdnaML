from typing import List
from torch import nn
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils import layers, locate_class
import torch

# CHANGELOG: secondary attention is List, not a single number


class ClassificationResnet(ModelAbstract):
    """Basic CoLabel Resnet model.

    A CoLabel model is a base ResNet, but during prediction, employs additional pieces such as 
    an ensemble voter, a heuristic based on the prediction output probabilities, as well as (if desired), 
    holistic nested side inputs.

    Args: (TODO)
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.
        softmax_dimensions (int, None): Number of FC dimensions for classification

    Kwargs (MODEL_KWARGS):
        last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
        attention (str, None): The attention module to use. Only supports ['cbam', None]
        input_attention (bool, false): Whether to include the IA module
        secondary_attention (List[int], None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic blocks provided here in List.
        part_attention (bool): Whether to use local attention

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

    model_name = "ClassificationResNet"
    model_arch = "ClassificationResNet"
    number_outputs = 1
    output_classnames = ["out1"]
    softmax_dimensions = [2048]
    secondary_outputs = []

    def __init__(
        self, base="resnet50", weights=None, normalization=None, metadata=None, parameter_groups: List[str]=None, **kwargs
    ):
        super().__init__(
            base=base,
            weights=weights,
            normalization=normalization,
            metadata=metadata,
            parameter_groups=parameter_groups,
            **kwargs
        )

    def model_attributes_setup(self, **kwargs):
        self.embedding_dimensions = kwargs.get("embedding_dimensions", None)
        if self.normalization == "":
            self.normalization = None

        self.softmax_dimensions = [
            kwargs.get("softmax_dimensions", self.metadata.getLabelDimensions())
        ]

        self.output_classnames = [kwargs.get("output_classnames", "out1")]

        self.base = None
        self.gap = None
        self.emb_linear = None
        self.feat_norm = None
        self.softmax = None

    def model_setup(self, **kwargs):
        self.build_base(
            **kwargs
        )  # All kwargs are passed into build_base,, which in turn passes kwargs into _resnet()
        self.build_normalization()
        self.build_softmax()

    def build_base(self, **kwargs):
        """Build the model base.

        Builds the architecture base/core.
        """
        _resnet = locate_class(subpackage="backbones", classpackage=self.model_base, classfile="resnet")
        # Set up the resnet backbone
        self.base = _resnet(last_stride=1, **kwargs)
        if self.weights is not None:
            self.base.load_param(self.weights)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.emb_linear = torch.nn.Identity()
        if self.embedding_dimensions is None:
            self.embedding_dimensions = 512 * self.base.block.expansion
        if self.embedding_dimensions != 512 * self.base.block.expansion:
            self.emb_linear = nn.Linear(
                self.base.block.expansion * 512, self.embedding_dimensions, bias=False
            )

    def build_normalization(self):
        if self.normalization == "bn":
            self.feat_norm = nn.BatchNorm1d(self.embedding_dimensions)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "in":
            self.feat_norm = layers.FixedInstanceNorm1d(
                self.embedding_dimensions, affine=True
            )
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "gn":
            self.feat_norm = nn.GroupNorm(
                self.embedding_dimensions // 16, self.embedding_dimensions, affine=True
            )
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "ln":
            self.feat_norm = nn.LayerNorm(
                self.embedding_dimensions, elementwise_affine=True
            )
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "l2":
            self.feat_norm = layers.L2Norm(self.embedding_dimensions, scale=1.0)
        elif self.normalization is None or self.normalization == "":
            self.feat_norm = torch.nn.Identity()
        else:
            raise NotImplementedError()

    def build_softmax(self, **kwargs):
        if self.softmax_dimensions[0] is not None:
            self.softmax = nn.Linear(
                self.embedding_dimensions, self.softmax_dimensions[0], bias=False
            )
            self.softmax.apply(self.weights_init_softmax)
        else:
            self.softmax = None  # TODO replace this with a zero compute layer that yields zero and has no_grad...

    def base_forward(self, x):
        features = self.gap(self.base(x))
        features = features.view(features.shape[0], -1)
        features = self.emb_linear(features)
        return features

    def forward_impl(self, x, **kwargs):
        features = self.base_forward(x)

        # if self.feat_norm is not None: <-- no need, identity
        features = self.feat_norm(features)

        soft_logits = None
        if self.softmax:
            soft_logits = self.softmax(features)
        return (
            soft_logits,
            features,
            [],
        )  # soft logits are the softmax logits we will use to for training. We can use features to store the historical probability????

    def parameter_groups_setup(self, parameter_groups: List[str]):
        self.parameter_groups[parameter_groups[0]] = self