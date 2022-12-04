from ednaml.metrics.BaseMetric import BaseMetric
import torchmetrics

class BaseTorchMetric(BaseMetric):
    """Wrapper class for TorchMetrics metrics for use in EdnaML."""
    def __init__(self, metric_name, torch_metric, metric_args, metric_params):
        self.metric = torch_metric
        self.metric_args = metric_args
        self.metric_params = metric_params
        self.metric_obj = None
        self.required_params = []
        super().__init__(metric_name, metric_type='EdnaML_TorchMetric')

    def build_module(self):
        # Define Metric Object
        self.metric_obj = self.metric(**self.metric_args) if self.metric_args else self.metric()
        # Itemize list of required parameters for update(). Used to pick up all relevant params from update() kwargs
        for param in self.metric_params.keys():
            self.required_params.append(param)

    def post_init_val(self):
        print('post-init:', type(self.metric))
        pass#assert isinstance(self.metric, torchmetrics.Metric), 'The provided metric object is not a TorchMetric.'

    def update(self,epoch,**kwargs):
        raise NotImplementedError