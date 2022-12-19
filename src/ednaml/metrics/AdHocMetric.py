
from ednaml.metrics import BaseMetric


class AdHocMetric(BaseMetric):

    def build_metric(self, **kwargs):
        pass

    def post_init_val(self, **kwargs):
        pass

    def _compute_metric(self, epoch, step, metric_name, metric_val, metric_type) -> float:
        return (metric_name, metric_type, "adhoc", epoch, step, metric_val)

    

