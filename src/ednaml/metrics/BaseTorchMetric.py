from ednaml.metrics.BaseMetric import BaseMetric
import torchmetrics

class BaseTorchMetric(BaseMetric):
    """Wrapper class for TorchMetrics metrics for use in EdnaML."""
    def __init__(self, metric_name, torch_metric, metric_params):
        self.metric = torch_metric
        self.metric_params=metric_params
        self.required_args = []
        self.metric_obj = None
        super().__init__(metric_name, metric_type='EdnaML_TorchMetric')

    def build_module(self, **kwargs):
        # Define Metric Object
        self.metric_obj = self.metric(**self.metric_params) if self.metric_params else self.metric()
        # Itemize list of required kwargs
        print(f'Setting metric params: {self.metric_params}')
        for arg in self.metric_params.keys():
            self.required_args.append(arg)

    def post_init_val(self):
        pass#assert isinstance(self.metric, torchmetrics.Metric), 'The provided metric object is not a TorchMetric.'

    def update(self,epoch,**kwargs):
        raise NotImplementedError