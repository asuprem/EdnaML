from os import PathLike
from torch import TensorType, nn
import torch
from typing import Any, List, Dict, Tuple, Type
from ednaml.plugins import ModelPlugin
from ednaml.utils.LabelMetadata import LabelMetadata


class ModelAbstract(nn.Module):
    """ModelAbstract is the base class for an Edna Model used in EdnaML, EdnaDeploy, and any other Edna frameworks. ModelAbstract allows design of practically any neural-network based trainable model by inheriting from torch.nn.

    ModelAbstract allows users to define a model by following specific recipes. While reducing some flexibibility, the end result is to allow for better interoperability and extensibility.

    Attributes:
        model_name (str): The name for this model. Useful during debugging.
        model_arch (TODO): TODO
        number_outputs (int): The number of outputs for this model
        output_classnames (List[str]): The name for each output for this model. This can be set manually or programatically.
        output_dimensions (List[int]): The dimensions for each output for this model. This can be set manually or programatically.
        secondary_outputs (TODO): TODO
        model_base (str): Name for the core model architecture, of this is a modular construction. Useful during debugging.
        weights (str): Path to weights file for this model
        normalization (str): Name for type of normalization used in this model. Provided here since it is a common model characteristic.
        metadata (LabelMetadata): A LabelMetadata object for this model that provides information on the training input and mopdel output. 
        parameter_groups (Dict[str, nn.Module]): A named dictionary of a ModelAbstracts sub-components. For example, a GAN can have encoder, decoder, and discriminator parameter groups, if they are trained separately.

    Methods:
        _type_: _description_
    """
    model_name: str = "ModelAbstract"
    model_arch: str = None
    number_outputs: int = 1
    output_classnames: List[str] = ["out1"]
    output_dimensions: List[int] = [512]
    secondary_outputs: List[Any] = []

    model_base: str
    weights: str
    normalization: str
    metadata: LabelMetadata
    parameter_groups: Dict[str, nn.Module]

    plugins: Dict[str,ModelPlugin] = {}
    plugin_count: int = 0
    has_plugins: bool = False

    def __init__(
        self,
        base: str = None,
        weights: PathLike = None,
        metadata: LabelMetadata = None,
        normalization: str = None,
        parameter_groups: List[str] = None,
        **kwargs
    ):
        super().__init__()
        self.metadata: LabelMetadata = metadata
        self.model_base = base
        self.weights = weights
        self.normalization = normalization
        self.parameter_groups = {}
        self.inferencing = False

        self.model_attributes_setup(**kwargs)
        self.model_setup(**kwargs)
        self.parameter_groups_setup(parameter_groups)

    def model_attributes_setup(self, **kwargs):
        raise NotImplementedError()

    def model_setup(self, **kwargs):
        raise NotImplementedError()

    def parameter_groups_setup(self, parameter_groups: List[str]):
        self.parameter_groups[parameter_groups[0]] = "self"

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
        """Initialize linear weights to standard normal. Mean 0. Standard Deviation 0.001"""
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

    def forward(self, x, labels=None, **kwargs):    # TODO Labels for the plugins...?????
        if self.training and self.inferencing:
            raise ValueError(
                "Cannot inference and train at the same time! Call"
                " deinference() first, before train()"
            )

        # TODO, we neede a pre-forward hook, a forward hook, and a post foward hook
        if self.has_plugins:
            x, kwargs, secondary_output_queue_pre = self.pre_forward_hook(x, **kwargs)
        feature_logits, features, secondary_outputs = self.forward_impl(
            x, **kwargs
        )
        if self.has_plugins:
            feature_logits, features, secondary_outputs, secondary_output_queue_post = self.post_forward_hook(x, feature_logits, features, secondary_outputs, **kwargs)

        # TODO deal with secondary_output_queues
        if self.has_plugins:
            secondary_outputs += tuple(secondary_output_queue_pre)
            secondary_outputs += tuple(secondary_output_queue_post)
        return feature_logits, features, secondary_outputs

    def foward_impl(self, x, **kwargs) -> Tuple[TensorType,TensorType,List[Any]]:

        raise NotImplementedError()

    def getModelName(self):
        return self.model_name

    def getModelBase(self):
        return self.model_base

    def getModelArch(self):
        return self.model_arch

    def getParameterGroup(self, key: str) -> nn.Module:
        if (
            self.parameter_groups[key] == "self"
        ):  # avoid recursion bug, I think?
            return self
        else:
            return self.parameter_groups[key]

    def inference(self):
        self.eval()
        self.inferencing = True

    def deinference(self):
        self.inferencing = False

    """
    def train(self, mode: bool = True):
        self.inferencing = False
        return super().train(mode)

    def eval(self):
        self.inferencing = False
        return super().eval()
    """

    def convertForInference(self) -> "ModelAbstract":
        raise NotImplementedError

    #-------------------------------------------------------------------------------------------
    # Model Plugins and Hooks architecture
    def loadPlugins(self, plugin_path: PathLike, ignore_plugins: List[str] = []):
        plugin_dict = torch.load(plugin_path)
        for plugin_name in plugin_dict:
            if plugin_name in self.plugins:
                if plugin_name not in ignore_plugins:
                    self.plugins[plugin_name].load(plugin_dict[plugin_name])
                    print("Loading plugin with name %s"%plugin_name)
                else:
                    print("Ignoring plugin with name %s"%plugin_name)
            else:
                print("No plugin exists for name %s"%plugin_name)

    def savePlugins(self):
        save_dict = {}
        for plugin_name in self.plugins:
            save_dict[plugin_name] = self.plugins[plugin_name].save()

        return save_dict

    def addPlugin(self, plugin: Type[ModelPlugin], plugin_name: str = None, plugin_kwargs: Dict[str, Any] = {}):
        if plugin_name is None:
            plugin_name = plugin.name
        if plugin_name == "ModelPlugin":
            # TODO NOTE we are badly changing the ModelPlugin name to the class name
            raise ValueError("Potentially no actual plugin passed!")

        if plugin_name in self.plugins:
            raise KeyError("`plugin_name` %s already exists in self.plugins:  "%plugin_name)
        else:
            self.plugins[plugin_name] = plugin(**plugin_kwargs)
        
        self.plugin_count = len(self.plugins)
        self.has_plugins = self.plugin_count > 0

    def pre_epoch_hook(self, epoch: int = 0):
        for plugin in self.plugins:
            self.plugins[plugin].pre_epoch(model = self, epoch=epoch)

    def post_epoch_hook(self, epoch: int = 0):
        for plugin in self.plugins:
            self.plugins[plugin].post_epoch(model = self, epoch=epoch)

    def pre_forward_hook(self, x, **kwargs):
        # For example, L-Score will perturb the input, then itself call self.forward_impl with the perturbed input (i.e. itself very 
        # hacky under our own framework), then provide results to secondary_outputs
        secondaries: Dict[str, Any] = {}
        for plugin in self.plugins:
            x, kwargs, secondary_output_pre = self.plugins[plugin].pre_forward(x, **kwargs)
            secondaries[plugin] = secondary_output_pre
        return x, kwargs, secondaries

    def post_forward_hook(self, x, feature_logits, features, secondary_outputs, **kwargs):
        # For ModelPlugins.
        # For example, KMP, after forward pass, provides distance to nearest proxy as well as nearest proxy in the secondary_outputs
        secondaries: Dict[str, Any] = {}
        for plugin in self.plugins:
            feature_logits, features, secondary_outputs, kwargs, secondary_output_post = self.plugins[plugin].post_forward(x,feature_logits, features, secondary_outputs,**kwargs)
            secondaries[plugin] = secondary_output_post
        return feature_logits, features, secondary_outputs, secondaries
