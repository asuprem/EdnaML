import torch
class BaseOptimizer:
    """ Base Optimizer Builder

    """
    name:str="optimizer1"
    def __init__(self,base_lr, lr_bias, gpus, weight_decay, weight_bias):
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
        self.base_lr = base_lr
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.lr_bias = lr_bias
        self.weight_bias = weight_bias


    def build(self, model, name = 'Adam', **kwargs) -> torch.optim.Optimizer:
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
                params += [{"params": [value], "lr":learning_rate, "weight_decay": weight_decay}]
        optimizer = __import__('torch.optim', fromlist=['optim'])
        optimizer = getattr(optimizer, name)
        optimizer = optimizer(params, **kwargs)
        return optimizer  


from .ClassificationOptimizer import ClassificationOptimizer
from .StandardLossOptimizer import StandardLossOptimizer
# for re-id compatibility 