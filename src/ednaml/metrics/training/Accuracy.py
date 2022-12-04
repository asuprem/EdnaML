"""
Module containing training metrics for use in EdnaML
"""

from ednaml.metrics import BaseTorchMetric
from torchmetrics import Accuracy as Torch_Accuracy

class TorchAccuracyMetric(BaseTorchMetric):
    def __init__(self,metric_name, metric_args, metric_params):
        super().__init__(metric_name,Torch_Accuracy,metric_args=metric_args,metric_params=metric_params)

    def update(self,epoch,**kwargs):
        print('kwargs in update:')
        print(kwargs)
        # Pick up the kwargs needed for update() from the full set of kwargs
        required_kwargs = {param: value for param, value in kwargs.items() if param in self.required_params}
        result = self.metric_obj(**required_kwargs)
        # Save metric state
        self.save(epoch, result)
