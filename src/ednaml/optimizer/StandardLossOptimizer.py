import importlib
import torch
from ednaml.loss.builders import LossBuilder
from ednaml.optimizer import BaseOptimizer
from ednaml.utils import locate_class


class StandardLossOptimizer(BaseOptimizer):
    """ Optimizer for Loss Functions

    This sets up optimizer for the loss functions in a LossBuilder. Not applicable everywhere. Not used everywhere because most losses will have zero differentiable parametersif 
    
    However, it will be useful for losses like the ProxyNCA, which need to learn proxies during training.

    """

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
        super(StandardLossOptimizer, self).__init__(
            name,
            optimizer,
            base_lr,
            lr_bias,
            gpus,
            weight_decay,
            weight_bias,
            opt_kwargs,
        )

    def build(self, loss_builder: LossBuilder) -> torch.optim.Optimizer:
        """ Builds an optimizer.

        Args:
        loss_builder (loss.builders.LossBuilder): A LossBuilder object
        name (str): name of torch.optim object to build
        kwargs (dict): any parameters that need to be passed into the optimizer

        Returns:
        torch.optim object

        """
        params = []
        for key, value in loss_builder.named_parameters():
            if value.requires_grad:
                # if "bias" in key:
                #    learning_rate = self.base_lr * self.lr_bias
                #    weight_decay = self.weight_decay * self.weight_bias
                # else:
                learning_rate = self.base_lr * self.gpus
                weight_decay = self.weight_decay
                params += [
                    {
                        "params": [value],
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                    }
                ]
        if len(params) == 0:
            return None
        optimizer = locate_class("torch","optim",self.optimizer)
        optimizer = optimizer(params, **self.kwargs)
        return optimizer
