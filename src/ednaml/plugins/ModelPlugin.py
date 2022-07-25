from typing import Any
from torch import nn



class ModelPlugin(nn.Module):
    name: str = "ModelPlugin"
    def __init__(self, **kwargs):
        
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

    def pre_epoch(self, epoch: int = 0, **kwargs):
        """Hook that is executed during the model training by BaseTrainer. Occurs before an epoch starts

        Args:
            epoch (int, optional): _description_. Defaults to 0.
        """
        pass

    def post_epoch(self, epoch: int = 0, **kwargs):
        """Hook that is executed during model training by BaseTrainer. Occurs at the end of an epoch.

        Args:
            epoch (int, optional): _description_. Defaults to 0.
        """
        pass