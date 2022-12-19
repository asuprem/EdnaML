"""
Module containing training metrics for use in EdnaML
"""

from ednaml.metrics import BaseTorchMetric

class TorchAccuracyMetric(BaseTorchMetric):

    def build_metric(self, **kwargs):
        metric_name = "Accuracy"
        self.metric = self.instantiate_metric(metric_name, kwargs)


    def update(self,epoch,batch,**kwargs):
        # Pick up the kwargs needed for update() from the full set of kwargs
        params_dict = {key:kwargs(self.params_dict[key]) for key in self.params_dict}
        # Custom result handling, per-metric basis. Result must be JSON-serializable
        result = self.metric(**params_dict).item()
        self.save(epoch, batch, result)

class TorchF1ScoreMetric(BaseTorchMetric):
    def __init__(self,metric_name, metric_args, metric_params):
        super().__init__(metric_name,Torch_F1Score,metric_args=metric_args,metric_params=metric_params)

    def update(self,epoch,batch,**kwargs):
        # Pick up the kwargs needed for update() from the full set of kwargs
        required_kwargs = {param: value for param, value in kwargs.items() if param in self.required_params}
        # Custom result handling, per-metric basis. Result must be JSON-serializable
        result = self.metric_obj(**required_kwargs).item()
        # Save metric state
        self.save(epoch, batch, result)
