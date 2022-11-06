"""
Module containing training metrics for use in EdnaML
"""

from ednaml.metrics import BaseTorchMetric
from torchmetrics import Accuracy as Torch_Accuracy

class TorchAccuracyMetric(BaseTorchMetric):
    def __init__(self,metric_name, metric_args):
        super().__init__(metric_name,Torch_Accuracy,metric_args)
    def build_module(self):
        print(f'Pretending to build TorchAccuracyMetric called {self.metric_name}')
        self.metric_obj = self.metric(**self.metric_args) if self.metric_args else self.metric()
    def update(self,**kwargs):
        required_args = {'preds': kwargs['preds'], 'target': kwargs['target']}
        result = self.metric_obj(required_args)
        self.results.append(result)