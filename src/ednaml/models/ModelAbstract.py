from torch import nn
import torch
from typing import List, Dict
from ednaml.utils.LabelMetadata import LabelMetadata


class ModelAbstract(nn.Module):
    model_name = "ModelAbstract"
    model_arch = None
    number_outputs = 1
    output_classnames = ["out1"]
    output_dimensions = [512]
    secondary_outputs = []

    model_base: str
    weights: str
    normalization: str
    metadata: LabelMetadata
    parameter_groups: Dict[str,nn.Module]


    def __init__(
        self,
        base=None,
        weights=None,
        metadata: LabelMetadata = None,
        normalization: str=None,
        parameter_groups: List[str]=None,
        **kwargs
    ):
        super().__init__()
        self.metadata: LabelMetadata = metadata
        self.model_base = base
        self.weights = weights
        self.normalization = normalization
        self.parameter_groups = {}

        self.model_attributes_setup(**kwargs)
        self.model_setup(**kwargs)
        self.parameter_groups_setup(parameter_groups)

    def model_attributes_setup(self, **kwargs):
        raise NotImplementedError()

    def model_setup(self, **kwargs):
        raise NotImplementedError()

    def parameter_groups_setup(self, parameter_groups: List[str]):
        raise NotImplementedError()

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            nn.init.constant_(m.bias, 0.0)
        elif classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("BatchNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("GroupNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("LayerNorm") != -1:
            # if m.affine:
            #    nn.init.constant_(m.weight, 1.0)
            #    nn.init.constant_(m.bias, 0.0)
            pass
        elif classname.find("InstanceNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_softmax(self, m):
        """ Initialize linear weights to standard normal. Mean 0. Standard Deviation 0.001 """
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

    def partial_load(self, weights_path):
        params = torch.load(weights_path)
        for _key in params:
            if (
                _key not in self.state_dict().keys()
                or params[_key].shape != self.state_dict()[_key].shape
            ):
                continue
            self.state_dict()[_key].copy_(params[_key])

    def forward(self, x, **kwargs):
        feature_logits, features, secondary_outputs = self.forward_impl(x, **kwargs)

        return feature_logits, features, secondary_outputs

    def foward_impl(self, x, **kwargs):

        raise NotImplementedError()

    def getModelName(self):
        return self.model_name

    def getModelBase(self):
        return self.model_base

    def getModelArch(self):
        return self.model_arch

    def getParameterGroup(self, key:str)->nn.Module:
        return self.parameter_groups[key]
