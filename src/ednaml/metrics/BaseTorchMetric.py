from ednaml.metrics.BaseMetric import BaseMetric
import torchmetrics

class BaseTorchMetric(BaseMetric):
    """Wrapper class for TorchMetrics metrics for use in EdnaML."""
    def __init__(self, metric_name, torch_metric, metric_args):
        super().__init__(metric_name, metric_type='EdnaML_TorchMetric')
        self.metric = torch_metric
        self.metric_args = metric_args
        self.metric_obj = None
        self.results = None

    def build_module(self, **kwargs):
        self.metric_obj = self.metric(**self.metric_args) if self.metric_args else self.metric()
        self.results = []

    def build_params(self, **kwargs):
        pass # TODO how to implement? How do we know which member objects eg. depth for sklearn tree

    def post_init_val(self):
        assert isinstance(self.metric_obj, torchmetrics.Metric), 'The provided metric object is not a TorchMetric.'

    def update(self, **kwargs):
        result = self.metric_obj(**kwargs)
        self.results.append(result)