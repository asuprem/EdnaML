
from typing import Any, Dict
from ednaml.metrics import BaseMetric


class AdHocMetric(BaseMetric):

    def build_metric(self, **kwargs):
        pass

    def post_init_val(self, **kwargs):
        pass

    def update(self, epoch: int, step: int, params: Dict[str, Any]) -> bool:
        self.memory.append(
            (params["metric_name"], params["metric_type"], params["metric_class"], epoch, step, params["metric_val"])
        )

        return True

    # TODO need to implement serialization code as well...


    

