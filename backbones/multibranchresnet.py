from ctypes import Union
import warnings
from torch import nn
import torch
import pdb

from utils.blocks import ChannelAttention, SpatialAttention
from utils.blocks import DenseAttention, InputAttention
from utils.blocks import ResnetInput, ResnetBasicBlock, ResnetBottleneck



class multibranchresnet(nn.Module):
    def __init__(self, block=ResnetBottleneck, layers=[3, 4, 6, 3], last_stride=2, zero_init_residual=False, \
                    top_only=True, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None,norm_layer=None, 
                    attention=None, input_attention = None, secondary_attention=None, ia_attention = None, part_attention = None,
                    num_branches=2, shared_block=0,
                    **kwargs):
        super().__init__()
        
        self.pytorch_weights_paths = self._model_weights()
        self.block=block
        self.inplanes = 64
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        #elif norm_layer == "ln":
        #    self._norm_layer = nn.LayerNorm
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be `None` or a 3-element tuple. Got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Attention parameters
        self.attention=attention
        self.input_attention=input_attention
        self.ia_attention = ia_attention
        self.part_attention = part_attention
        self.secondary_attention=secondary_attention
        
        # Make sure ia and input_attention do not conflict
        if self.ia_attention is not None and self.input_attention is not None:
            raise ValueError("Cannot have both ia_attention and input_attention.")
        if self.part_attention is not None and (self.attention is not None and self.secondary_attention is None):
            raise ValueError("Cannot have part-attention with CBAM everywhere")
        if self.part_attention is not None and (self.attention is not None and self.secondary_attention==1):
            raise ValueError("Cannot have part-attention with CBAM-Early")

        # Here, set up where the branching begins in the resnet backbone
        self.shared_block_count = shared_block
        if self.shared_block_count>4:
            raise ValueError("`shared_block_count` value is %i. Cannot be greater than 4"%self.shared_block_count)
        if self.shared_block_count==4:
            raise ValueError("`shared_block_count` value is %i. This is a non-branching model."%self.shared_block_count)

        # Set up the per-layer parameters for the primary ResNet blocks
        layer_strides = [1,2,2,last_stride]
        layer_part_attention = [self.part_attention, False, False, False]
        layer_input_attention = [self.input_attention, False, False, False]
        layer_dilate = [False]+replace_stride_with_dilation
        layer_outplanes = [64,128,256,512]
        # Fix secondary attention
        if secondary_attention is None:
            layer_att = [self.attention]*4
        else:
            layer_att=[None]*4
            layer_att[secondary_attention] = self.attention
        # Zip layer arguments
        layer_arguments = list(zip(layers, layer_outplanes, layer_strides, layer_part_attention, layer_input_attention, layer_dilate, layer_att))
        
        # First, given the shared_block_count, generate the shared layers list. We will nn.Sequential them later
        sharedlayers=[]
        for layer_zip in layer_arguments[:self.shared_block_count]:
            sharedlayers.append(
                self._make_layer(   self.block, 
                                    layer_zip[1], 
                                    layer_zip[0], 
                                    attention=layer_zip[6],
                                    input_attention=layer_zip[4],
                                    part_attention=layer_zip[3],
                                    dilate=layer_zip[5],
                                    stride=layer_zip[2])
            )

        # Then, given the branches, put the remaining resnet blocks in their branches
        # During prediction, we will just get branch features so order does not matter yet. it will matter in MultiBranchResnet
        # So, self.branches will be a nn.moduleList, with a bunch of nn.Sequentials
        self.num_branches = num_branches
        branches = [None]*self.num_branches
        for bidx in range(self.num_branches):
            branches[bidx] = []
            for layer_zip in layer_arguments[self.shared_block_count:]:
                branches[bidx].append(
                    self._make_layer(   self.block, 
                                        layer_zip[1], 
                                        layer_zip[0], 
                                        attention=layer_zip[6],
                                        input_attention=layer_zip[4],
                                        part_attention=layer_zip[3],
                                        dilate=layer_zip[5],
                                        stride=layer_zip[2])
                )
            branches[bidx] = nn.Sequential(*branches[bidx])

        self.resnetinput = ResnetInput(ia_attention=ia_attention)
        if len(sharedlayers)>0:
            self.sharedblock = nn.Sequential(*sharedlayers)
        else:
            self.sharedblock = nn.Identity()
        self.branches = nn.ModuleList(branches)
        
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate = False, attention = None, input_attention=False, ia_attention = False, part_attention = False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,groups = self.groups, base_width = self.base_width, dilation = previous_dilation, norm_layer=self._norm_layer, attention=attention, input_attention=input_attention, part_attention=part_attention))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups = self.groups, base_width = self.base_width, dilation = self.dilation, norm_layer=self._norm_layer, attention=attention))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.resnetinput(x)
        x = self.sharedblock(x)
        return [self.branches[idx](x) for idx in range(self.num_branches)]
    
    def load_param(self, weights_path):
        if weights_path in self.pytorch_weights_paths:
            self.load_params_from_pytorch(weights_path)
        else:
            self.load_params_from_weights(weights_path)

    def load_params_from_pytorch(self, weights_path):
        pass

    

    def load_params_from_weights(self, weights_path):
        param_dict = torch.load(weights_path)
        for i in param_dict:
            if 'fc' in i and self.top_only:
                continue
            self.state_dict()[i].copy_(param_dict[i])


    def _model_weights(self):
        mw = ['resnet18-5c106cde.pth', 
                'resnet34-333f7ec4.pth', 
                'resnet50-19c8e357.pth', 
                'resnet101-5d3b4d8f.pth', 
                'resnet152-b121ed2d.pth', 
                'resnext50_32x4d-7cdf4587.pth', 
                'resnext101_32x8d-8ba56ff5.pth', 
                'wide_resnet50_2-95faca4d.pth', 
                'wide_resnet50_2-95faca4d.pth', 
                'resnet18-5c106cde_cbam.pth', 
                'resnet34-333f7ec4_cbam.pth', 
                'resnet50-19c8e357_cbam.pth', 
                'resnet101-5d3b4d8f_cbam.pth', 
                'resnet152-b121ed2d_cbam.pth']
        return {item:1 for item in mw}



def _multibranchresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = multibranchresnet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet('resnet18', ResnetBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet('resnet34', ResnetBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet('resnet50', ResnetBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet('resnet101', ResnetBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _multibranchresnet('resnet152', ResnetBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _multibranchresnet('resnext50_32x4d', ResnetBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _multibranchresnet('resnext101_32x8d', ResnetBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


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
    kwargs['width_per_group'] = 64 * 2
    return _multibranchresnet('wide_resnet50_2', ResnetBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


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
    kwargs['width_per_group'] = 64 * 2
    return _multibranchresnet('wide_resnet101_2', ResnetBottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
