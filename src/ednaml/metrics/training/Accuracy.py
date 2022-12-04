"""
Module containing training metrics for use in EdnaML
"""

from ednaml.metrics import BaseTorchMetric
from torchmetrics import Accuracy as Torch_Accuracy

class TorchAccuracyMetric(BaseTorchMetric):
    def __init__(self,metric_name, metric_args, metric_params):
        super().__init__(metric_name,Torch_Accuracy,metric_args=metric_args,metric_params=metric_params)

    def update(self,epoch,**kwargs):
        required_params = {arg: value for arg, value in kwargs.items() if arg in self.metric_params}
        result = self.metric_obj(required_params)
        self.state[self.metric_type][self.metric_name][epoch] = result