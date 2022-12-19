from ednaml.metrics.BaseMetric import BaseMetric
import json
from ednaml.utils import locate_class

class BaseTorchMetric(BaseMetric):
    """Wrapper class for TorchMetrics metrics for use in EdnaML."""
    """
    def __init__(self, metric_name, torch_metric, metric_args, metric_params):
        self.metric = torch_metric
        self.metric_args = metric_args
        self.metric_params = metric_params
        self.metric_obj = None
        self.required_params = []
        super().__init__(metric_name, metric_type='EdnaML_TorchMetrics')
    """
    def build_metric(self, **kwargs):
        """Torchmetric wrappers should implement this
        TODO: Check whether torchmetrics is fully consistent
        if so, we may be able to directly use BaseTorchMetric for everything...!

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def instantiate_metric(self, metric_name, metric_kwargs):
        metric_class = locate_class(package="torchmetrics", subpackage="metric_name")
        return metric_class(**metric_kwargs)


    def post_init_val(self):
        pass
    
    # TODO
    def save(self, epoch, batch, result):
        # Ensure the result is JSON-serializable
        try:
            json.dumps(result)
        except TypeError:
            raise ValueError(f'The computed result of the EdnaML_TorchMetric \' {self.metric_name} \' ({result}) is not JSON-serializable.')
        # Key=Type-Metric-Epoch-Batch, Value=Result
        self.state[self.metric_type][self.metric_name][epoch] = {batch: result}