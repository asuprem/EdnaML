"""Contains code to build a resnet

  Typical usage example:

  model = resnet()
  x = torch.randn((3,100,100))
  features = model(x)
"""

from typing import List
from torch import nn
import torch

from ednaml.utils.blocks import InputAttention
from ednaml.utils.blocks import ResnetBasicBlock as BasicBlock
from ednaml.utils.blocks import ResnetBottleneck as Bottleneck


class resnet(nn.Module):
    def __init__(
        self,
        block: nn.Module = Bottleneck,
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
        **kwargs
    ):
        """Initializes the resnet model.


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

        Raises:
            ValueError: If attention blocks are not self-consistent. Specifically, the following rules:
                - cannot have both `ia_attention` and `input_attention`.
                - cannot have `part_attention` with `attention`, unless `secondary_attention`!=1
            ValueError: If `replace_stride_with_dilation` is not a 3-tuple or None.
        """
        super().__init__()

        self.attention = attention
        self.input_attention = input_attention
        self.secondary_attention = secondary_attention
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
                "replace_stride_with_dilation should be `None` or a 3-element tuple. Got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        # if norm_layer == "gn":
        #    self.bn1 = nn.GroupNorm2d
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ia_attention = ia_attention
        self.part_attention = part_attention

        # Make sure ia and input_attention do not conflict
        if self.ia_attention is not None and self.input_attention is not None:
            raise ValueError("Cannot have both ia_attention and input_attention.")
        if self.part_attention is not None and (
            self.attention is not None and self.secondary_attention is None
        ):
            raise ValueError("Cannot have part-attention with CBAM everywhere")
        if self.part_attention is not None and (
            self.attention is not None and self.secondary_attention == 1
        ):
            raise ValueError("Cannot have part-attention with CBAM-Early")

        # Create true IA
        if self.ia_attention:
            self.ia_attention = InputAttention(self.inplanes)  # 64, set above
        else:
            self.ia_attention = None

        att = self.attention
        if (
            secondary_attention is not None and secondary_attention != 1
        ):  # leave alone if sec attention not set
            att = None
        self.layer1 = self._make_layer(
            self.block,
            64,
            layers[0],
            attention=att,
            input_attention=self.input_attention,
            part_attention=self.part_attention,
        )
        att = self.attention
        if (
            secondary_attention is not None and secondary_attention != 2
        ):  # leave alone if sec attention not set
            att = None
        self.layer2 = self._make_layer(
            self.block,
            128,
            layers[1],
            stride=2,
            attention=att,
            dilate=replace_stride_with_dilation[0],
        )
        att = self.attention
        if (
            secondary_attention is not None and secondary_attention != 3
        ):  # leave alone if sec attention not set
            att = None
        self.layer3 = self._make_layer(
            self.block,
            256,
            layers[2],
            stride=2,
            attention=att,
            dilate=replace_stride_with_dilation[1],
        )
        att = self.attention
        if (
            secondary_attention is not None and secondary_attention != 4
        ):  # leave alone if sec attention not set
            att = None
        self.layer4 = self._make_layer(
            self.block,
            512,
            layers[3],
            stride=last_stride,
            attention=att,
            dilate=replace_stride_with_dilation[2],
        )

        self.top_only = top_only
        self.avgpool, self.fc = None, None

        if not self.top_only:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)

        if self.ia_attention is not None:
            x = self.ia_attention(x) * x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.top_only:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def load_param(self, weights_path: str):
        """Loads parameters frm saved weights file

        Args:
            weights_path (str): Path to the weights file
        """
        param_dict = torch.load(weights_path)
        for i in param_dict:
            if "fc" in i and self.top_only:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def _resnet(arch, block, layers, pretrained, progress, **kwargs) -> resnet:
    """Builds a resnet

    Args:
        arch (str): Architecture base. Unused.
        block (nn.Module): The class of the ResnetBlock (BasicBlock or Bottleneck)
        layers (List[int]): The layers for each ResnetBlock
        pretrained (bool): Unused
        progress (bool): Unused

    Returns:
        resnet: The `resnet` model
    """
    model = resnet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
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
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
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
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
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
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
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
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )
