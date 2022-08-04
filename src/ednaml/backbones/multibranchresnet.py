"""Contains code to build a multi-branch resnet, with a mix of
weight-shared and independent branches.

`multibranchresnet` creates subbranches contained all or part of resnet. 
Branching occurs at a specified resnet block, after which the remaining 
blocks occur on their own branches with no weight sharing. The branched 
model also has an option for branch fusing, to concatenate features.

  Typical usage example:

  model = multibranchresnet()
  x = torch.randn((3,100,100))
  features = model(x)
"""

import os
from torch import nn
import torch
from typing import Dict, List
from ednaml.utils.blocks import ResnetInput, ResnetBasicBlock, ResnetBottleneck


class multibranchresnet(nn.Module):
    """`multibranchresnet` creates a resnet with specified branches.

    `multibranchresnet` creates subbranches contained all or part of resnet.
    Branching occurs at a specified resnet block, after which the remaining
    blocks occur on their own branches with no weight sharing. The branched
    model also has an option for branch fusing, to concatenate features.

    Attributes:
        block (nn.Module): The block (BasicBlock or Bottleneck) used for the internal Resnet architecture
        attention (str): Which type of attentio, if any, is implemented in this model. One of `cbam`, `dbam`
        input_attention (bool): Whether model uses input attention at the first resnet block
        ia_attention (bool): Whether the model uses input attention at the first layer
        part_attention (bool): Whether the model uses local/part attention at the first resnet block
        secondary_attention (None | int): Whether `attention` is applied to all blocks (None) or to the specified block (int).
        shared_block_count (int): Number of resnet blocks with weight sharing. Maximum value 4
        number_branches (int): Number of branches after weight-shared blocks
        pytorch_weights_paths (Dict[str,int]): Strings corresponding to official imagenet resnet weights from pytorch
        resnetinput (ResnetInput): The input block consisting of conv layer, relu, and pooling.
        sharedblock (Union[nn.Sequential,nn.Identity]): The shared layers, consisting of at most 4 Resnet blocks.
        branches (nn.ModuleList): List of branching layers, with at most 4 Resnet blocks each.
    """

    block: nn.Module
    attention: str
    input_attention: bool
    ia_attention: bool
    part_attention: bool
    secondary_attention: int

    shared_block_count: int
    num_branches: int
    pytorch_weights_paths: Dict[str, int]

    resnetinput: ResnetInput
    sharedblock: nn.Sequential
    branches: nn.ModuleList

    def __init__(
        self,
        block: nn.Module = ResnetBottleneck,
        layers: List[int] = [3, 4, 6, 3],
        last_stride: int = 2,
        zero_init_residual: bool = False,
        top_only: bool = True,
        num_classes: bool = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: List[int] = None,
        norm_layer: nn.Module = None,
        attention: str = None,
        input_attention: bool = None,
        secondary_attention: int = None,
        ia_attention: bool = None,
        part_attention: bool = None,
        number_branches: int = 2,
        shared_block: int = 0,
        **kwargs
    ):
        """Initializes the multibranchresnet model and sets up internal modules.

        Args:
            block (nn.Module, optional): The block (BasicBlock or Bottleneck) used for the internal Resnet architecture. Defaults to ResnetBottleneck.
            layers (List[int], optional): Number of layers in each block. Defaults to [3, 4, 6, 3].
            last_stride (int, optional): The stride for the last block. Defaults to 2.
            zero_init_residual (bool, optional): Whether to initialize network with only zeros. Unused. Defaults to False.
            top_only (bool, optional): Whether to keep only the feature extractor block. Defaults to True.
            num_classes (bool, optional): Number of classes for the imagenet layer. Unused. Defaults to 1000.
            groups (int, optional): see nn.conv2D. Defaults to 1.
            width_per_group (int, optional): see nn.conv2D. Defaults to 64.
            replace_stride_with_dilation (List[int], optional): Whether to replace stride with dilation for each block. Defaults to None.
            norm_layer (nn.Module, optional): The default normalization layer. If None, uses batchnorm. Defaults to None.
            attention (str, optional): What attention to use, among `cbam`, `dbam`. Defaults to None.
            input_attention (bool, optional): Whether to use input attention at the first resnet block. Defaults to None.
            secondary_attention (int, optional): Whether to use secondary attention to apply `attention` to the specific resnet block only. Defaults to None.
            ia_attention (bool, optional): Whether to use input attention at the first conv layer. Exclusive with `input_attention`. Defaults to None.
            part_attention (bool, optional): Whether to use the local attention module. Defaults to None.
            num_branches (int, optional): Number of branches for this resnet. Defaults to 2.
            shared_block (int, optional): Number of weight-shared resnet blocks. Defaults to 0.

        Raises:
            ValueError: If attention blocks are not self-consistent. Specifically, the following rules:
                - cannot have both `ia_attention` and `input_attention`.
                - cannot have `part_attention` with `attention`, unless `secondary_attention`!=1
            ValueError: If branching does not occur, i.e. `shared_block`>=4
            ValueError: If `replace_stride_with_dilation` is not a 3-tuple or None.
        """
        super().__init__()

        self.pytorch_weights_paths = self._model_weights()
        self.block = block
        self.inplanes = 64
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        # elif norm_layer == "ln":
        #    self._norm_layer = nn.LayerNorm
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be `None` or a 3-element"
                " tuple. Got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # Attention parameters
        self.attention = attention
        self.input_attention = input_attention
        self.ia_attention = ia_attention
        self.part_attention = part_attention
        self.secondary_attention = secondary_attention

        # Make sure ia and input_attention do not conflict
        if self.ia_attention is not None and self.input_attention is not None:
            raise ValueError(
                "Cannot have both ia_attention and input_attention."
            )
        if self.part_attention is not None and (
            self.attention is not None and self.secondary_attention is None
        ):
            raise ValueError("Cannot have part-attention with CBAM everywhere")
        if self.part_attention is not None and (
            self.attention is not None and self.secondary_attention == 1
        ):
            raise ValueError("Cannot have part-attention with CBAM-Early")

        # Here, set up where the branching begins in the resnet backbone
        self.shared_block_count = shared_block
        if self.shared_block_count > 4:
            raise ValueError(
                "`shared_block_count` value is %i. Cannot be greater than 4"
                % self.shared_block_count
            )
        if self.shared_block_count == 4:
            raise ValueError(
                "`shared_block_count` value is %i. This is a non-branching"
                " model."
                % self.shared_block_count
            )

        # Set up the per-layer parameters for the primary ResNet blocks
        layer_strides = [1, 2, 2, last_stride]
        layer_part_attention = [self.part_attention, False, False, False]
        layer_input_attention = [self.input_attention, False, False, False]
        layer_dilate = [False] + replace_stride_with_dilation
        layer_outplanes = [64, 128, 256, 512]
        # Fix secondary attention
        if secondary_attention is None:
            layer_att = [self.attention] * 4
        else:
            layer_att = [None] * 4
            layer_att[secondary_attention] = self.attention
        # Zip layer arguments
        layer_arguments = list(
            zip(
                layers,
                layer_outplanes,
                layer_strides,
                layer_part_attention,
                layer_input_attention,
                layer_dilate,
                layer_att,
            )
        )

        # First, given the shared_block_count, generate the shared layers list. We will nn.Sequential them later
        sharedlayers = []
        for layer_zip in layer_arguments[: self.shared_block_count]:
            sharedlayers.append(
                self._make_layer(
                    self.block,
                    layer_zip[1],
                    layer_zip[0],
                    attention=layer_zip[6],
                    input_attention=layer_zip[4],
                    part_attention=layer_zip[3],
                    dilate=layer_zip[5],
                    stride=layer_zip[2],
                )
            )
        self.shared_inplanes = self.inplanes
        self.shared_dilation = self.dilation
        # Then, given the branches, put the remaining resnet blocks in their branches
        # During prediction, we will just get branch features so order does not matter yet. it will matter in MultiBranchResnet
        # So, self.branches will be a nn.moduleList, with a bunch of nn.Sequentials
        self.num_branches = number_branches
        branches = [None] * self.num_branches
        for bidx in range(self.num_branches):
            branches[bidx] = []
            for layer_zip in layer_arguments[self.shared_block_count :]:
                branches[bidx].append(
                    self._make_layer(
                        self.block,
                        layer_zip[1],
                        layer_zip[0],
                        attention=layer_zip[6],
                        input_attention=layer_zip[4],
                        part_attention=layer_zip[3],
                        dilate=layer_zip[5],
                        stride=layer_zip[2],
                    )
                )
            branches[bidx] = nn.Sequential(*branches[bidx])
            self.inplanes = self.shared_inplanes
            self.dilation = self.shared_dilation

        self.resnetinput = ResnetInput(ia_attention=ia_attention)
        if len(sharedlayers) > 0:
            self.sharedblock = nn.Sequential(*sharedlayers)
        else:
            self.sharedblock = nn.Identity()
        self.branches = nn.ModuleList(branches)

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        attention: str = None,
        input_attention: bool = False,
        ia_attention: bool = False,
        part_attention: bool = False,
    ) -> nn.Sequential:
        """Creates a resnet block

        Args:
            block (nn.Module): The block (BasicBlock or Bottleneck) used for the internal Resnet architecture. Defaults to ResnetBottleneck.
            planes (int): Number of input depth
            blocks (int): Number of blocks in this ResnetBlock
            stride (int, optional): Stride for the conv layers. Defaults to 1.
            dilate (bool, optional): Dilation for the conv layers. Defaults to False.
            attention (str, optional): Which of `cbam`, `dbam` attention to use. Defaults to None.
            input_attention (bool, optional): Whether to use `input_attention`. Defaults to False.
            ia_attention (bool, optional): Whether to use `ia_attention`. Unused. Defaults to False.
            part_attention (bool, optional): Whether to use local attention. Defaults to False.

        Returns:
            nn.Sequential: The layers comprising this Resnet Block as an nn.Sequential
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=self._norm_layer,
                attention=attention,
                input_attention=input_attention,
                part_attention=part_attention,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                    attention=attention,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnetinput(x)
        x = self.sharedblock(x)
        return [self.branches[idx](x) for idx in range(self.num_branches)]

    def load_param(self, weights_path: str):
        """Loads parameters from saved weights file

        Args:
            weights_path (str): Path to the weights file
        """
        if os.path.basename(weights_path) in self.pytorch_weights_paths:
            self.load_params_from_pytorch(weights_path)
        else:
            self.load_params_from_weights(weights_path)

    def load_params_from_pytorch(self, weights_path: str):
        """Loads default pytorch weights file into the multibranch resnet

        Args:
            weights_path (str): Path to the weights file
        """
        param_dict = torch.load(weights_path)
        # Three stages: load resnetinput params, load shared block params, and then load branch params...
        inputparams = [
            "conv1.weight",
            "bn1.running_mean",
            "bn1.running_var",
            "bn1.weight",
            "bn1.bias",
        ]
        # Load the input params
        for param in inputparams:
            self.state_dict()["resnetinput." + param].copy_(param_dict[param])

        # load shared block params,
        for layer_idx in range(self.shared_block_count):
            # layer_idx is the layer that is in shared-block. BUT, in the pytorch params, it exists as layer1, corresponding to layer_idx0
            # So if shared_block_count is 3, then we need to copy layer 1-3 into layer_idx0-2
            full_layer_list = [
                item
                for item in param_dict
                if ("layer" + str(layer_idx + 1) in item)
            ]  # get all weights inside layer[x]

            # The layers exist as an nn.sequential
            for layer_name in full_layer_list:
                # First, get the raw layer info, then append sharedblock to it...
                local_param_name = (
                    self._build_local_layer_param_from_pytorch_name(
                        layer_name, layer_idx
                    )
                )
                self.state_dict()[local_param_name].copy_(
                    param_dict[layer_name]
                )

        # Now, we need to load branch params...
        for layer_idx in range(self.shared_block_count, 4):
            # layer_idx is the layer that is in shared-block. BUT, in the pytorch params, it exists as layer1, corresponding to layer_idx0
            # So if shared_block_count is 3, then we need to copy layer 1-3 into layer_idx0-2
            full_layer_list = [
                item
                for item in param_dict
                if ("layer" + str(layer_idx + 1) in item)
            ]  # get all weights inside layer[x]

            # The layers exist as an nn.sequential
            for branch_idx in range(self.num_branches):
                for layer_name in full_layer_list:
                    # First, get the raw layer info, then append sharedblock to it...
                    local_param_name = (
                        self._build_branch_layer_param_from_pytorch_name(
                            layer_name,
                            layer_idx,
                            branch_idx,
                            self.shared_block_count,
                        )
                    )
                    self.state_dict()[local_param_name].copy_(
                        param_dict[layer_name]
                    )

    def _build_local_layer_param_from_pytorch_name(
        self, paramname: str, layeridx: int
    ) -> str:
        """Converts a parameter name from an unbranched pytorch model to the corresponding `multibranchresnet` version inside the shared block.

        Args:
            paramname (str): The original parameter name
            layeridx (int): The layer this parameter name originated from

        Returns:
            str: Converted parameter name
        """
        param_list = paramname.split(".")
        new_param_list = ["sharedblock", str(layeridx)] + param_list[1:]
        return ".".join(new_param_list)

    def _build_branch_layer_param_from_pytorch_name(
        self, paramname: str, layeridx: int, branch_idx: int, layer_reset: int
    ) -> str:
        """Converts a parameter name from an unbranched pytorch model to the corresponding `multibranchresnet` version inside the branches.

        Args:
            paramname (str): The original parameter name
            layeridx (int): The layer this parameter name originated from
            branch_idx (int): The target branch for this parameter
            layer_reset (int): The layer reset number, since `multibranchresnet` restarts layer numbering at the branch junction

        Returns:
            str: Converted parameter name
        """
        param_list = paramname.split(".")
        new_param_list = [
            "branches",
            str(branch_idx),
            str(layeridx - layer_reset),
        ] + param_list[1:]
        return ".".join(new_param_list)

    def load_params_from_weights(self, weights_path: str):
        """Loads parameters from a saved `multibranchresnet`

        Args:
            weights_path (str): Path to the weights file
        """
        param_dict = torch.load(weights_path)
        for i in param_dict:
            if "fc" in i and self.top_only:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def _model_weights(self) -> Dict[str, int]:
        """Constructs a dictionary to quickly retrieve pytorch weight paths

        Returns:
            Dict[str,int]: the dictinary storing paths
        """
        mw = [
            "resnet18-5c106cde.pth",
            "resnet34-333f7ec4.pth",
            "resnet50-19c8e357.pth",
            "resnet101-5d3b4d8f.pth",
            "resnet152-b121ed2d.pth",
            "resnext50_32x4d-7cdf4587.pth",
            "resnext101_32x8d-8ba56ff5.pth",
            "wide_resnet50_2-95faca4d.pth",
            "wide_resnet50_2-95faca4d.pth",
            "resnet18-5c106cde_cbam.pth",
            "resnet34-333f7ec4_cbam.pth",
            "resnet50-19c8e357_cbam.pth",
            "resnet101-5d3b4d8f_cbam.pth",
            "resnet152-b121ed2d_cbam.pth",
        ]
        return {item: 1 for item in mw}


def _multibranchresnet(
    arch, block, layers, pretrained, progress, **kwargs
) -> multibranchresnet:
    """Builds a `multibranchresnet`

    Args:
        arch (str): Architecture base. Unused.
        block (nn.Module): The class of the ResnetBlock (BasicBlock or Bottleneck)
        layers (List[int]): The layers for each ResnetBlock
        pretrained (bool): Unused
        progress (bool): Unused

    Returns:
        multibranchresnet: The `multibranchresnet` model
    """
    model = multibranchresnet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet(
        "resnet18",
        ResnetBasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        **kwargs
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet(
        "resnet34",
        ResnetBasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet(
        "resnet50",
        ResnetBottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet(
        "resnet101",
        ResnetBottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet(
        "resnet152",
        ResnetBottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _multibranchresnet(
        "resnext50_32x4d",
        ResnetBottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _multibranchresnet(
        "resnext101_32x8d",
        ResnetBottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _multibranchresnet(
        "wide_resnet50_2",
        ResnetBottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _multibranchresnet(
        "wide_resnet101_2",
        ResnetBottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )
