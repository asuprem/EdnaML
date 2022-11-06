"""
Module containing training metrics for use in EdnaML
"""

from endaml.metrics import BaseTorchMetric
from torchmetrics import Accuracy as Torch_Accuracy

class TorchAccuracyMetric(BaseTorchMetric):
    def __init__(self,metric_name, metric_args):
        super().__init__(metric_name,Torch_Accuracy,metric_args)
    def build_module(self):
        print(f'Pretending to build TorchAccuracyMetric called {self.metric_name}')
    def update(self,**kwargs):
        required_args = {'preds': kwargs['preds'], 'target': kwargs['target']}
        result = self.metric_obj(required_args)
        self.results.append(result)