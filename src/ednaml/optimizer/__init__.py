import importlib
import torch

from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils import locate_class


class BaseOptimizer:
    """ Base Optimizer Builder

    """

    name: str = "optimizer1"

    def __init__(
        self,
        name,
        optimizer,
        base_lr,
        lr_bias,
        gpus,
        weight_decay,
        weight_bias,
        opt_kwargs,
    ):
        """ Initializes the optimizer builder.

        Args:
        base_lr (float): Base learning rate for optimizer
        lr_bias (float): Multiplicative factor for bias parameters
        gpus (int): Number of GPUs for lr scaling
        weight_decay (float): Weight decay for decoupled weight decay optimizers like AdamW
        weight_bias (float): Multiplicative factor for bias parameters in weight decay optimizers

        Methods:
        build:  builds an optimizer given optimizer name and torch model

        """
        self.name = name
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.lr_bias = lr_bias
        self.weight_bias = weight_bias
        self.kwargs = opt_kwargs

    def build(self, model: ModelAbstract) -> torch.optim.Optimizer:
        """ Builds an optimizer.

        Args:
        model (torch.nn.Module): A model
        name (str): name of torch.optim object to build
        kwargs (dict): any parameters that need to be passed into the optimizer

        Returns:
        torch.optim object

        """
        params = []
        for key, value in model.named_parameters():
            if value.requires_grad:
                if "bias" in key:
                    learning_rate = self.base_lr * self.lr_bias
                    weight_decay = self.weight_decay * self.weight_bias
                else:
                    learning_rate = self.base_lr * self.gpus
                    weight_decay = self.weight_decay
                params += [
                    {
                        "params": [value],
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                    }
                ]
        optimizer = locate_class("torch","optim",self.optimizer)
        optimizer = optimizer(params, **self.kwargs)
        return optimizer


from ednaml.optimizer.ClassificationOptimizer import ClassificationOptimizer
from ednaml.optimizer.StandardLossOptimizer import StandardLossOptimizer

# for re-id compatibility
