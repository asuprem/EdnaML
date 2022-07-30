from typing import Any
from torch import nn

from ednaml.models.ModelAbstract import ModelAbstract



class ModelPlugin(nn.Module):
    name: str = "ModelPlugin"
    def __init__(self, **kwargs):
        super().__init__()
        self.build_plugin(**kwargs)
        self.build_params(**kwargs)
        
        
    def build_plugin(self, **kwargs):
        """Sets up plugin-specific variables
        """
        pass

    def build_params(self, **kwargs):
        """Sets up plugin operational parameters
        """
        pass

    def pre_forward(self, x, **kwargs):
        """Called before the forward pass of a ModelAbstract, to modify the inputs if needed and to add to the secondary outputs

        Args:
            x (_type_): _description_
        """
        return x, kwargs, {}

    def post_forward(self, x, feature_logits, features, secondary_outputs, **kwargs):
        """Called after the forward pass of a ModelAbstract, to use Plugin operations parameters to add to secondary outputs and adjust the outputs

        Args:
            x (_type_): _description_
            feature_logits (_type_): _description_
            features (_type_): _description_
            secondary_outputs (_type_): _description_
        """
        return feature_logits, features, secondary_outputs, kwargs, {}

    def pre_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        """Hook that is executed during the model training by BaseTrainer. Occurs before an epoch starts

        Args:
            epoch (int, optional): _description_. Defaults to 0.
        """
        pass

    def post_epoch(self, model: ModelAbstract, epoch: int = 0, **kwargs):
        """Hook that is executed during model training by BaseTrainer. Occurs at the end of an epoch.

        Args:
            epoch (int, optional): _description_. Defaults to 0.
        """
        pass

    def save(self):
        return self.__dict__
        # returns an object of itself for saving

    def load(self, save_dict_or_path):
        # TODO adjust this so plugin can be loaded from a file, instead of directly passing parameters...
        self.__dict__.update(save_dict_or_path) # probably need a better way to do this, i.e. iterate through dict, and keep what is ther, warn about extra keys, etc...