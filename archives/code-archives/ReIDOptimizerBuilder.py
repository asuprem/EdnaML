import torch
from . import BaseOptimizer


class ReIDOptimizerBuilder(BaseOptimizer):
    """ Optimizer Builder for ReID experiments.

    """
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
        super(ReIDOptimizerBuilder, self).__init__(base_lr, lr_bias, gpus, weight_decay, weight_bias)