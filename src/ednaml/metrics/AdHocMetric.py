
from typing import Any, Dict, Tuple, List
from ednaml.metrics import BaseMetric


class AdHocMetric(BaseMetric):

    def build_metric(self, **kwargs):
        pass

    def post_init_val(self, **kwargs):
        return (True, "")

    def update(self, epoch: int, step: int, params: Dict[str, Any]) -> bool:
        self.memory.append(
            (params["metric_name"], params["metric_type"], params["metric_class"], epoch, step, params["metric_val"])
        )
        # e.g. (loss, model, adhoc, 4, 100, 0.4)

        return True

    def _compute_metric(self, epoch: int, step: int, **kwargs) -> float:
        return super()._compute_metric(epoch, step, **kwargs)


    def _serialize_to_string(self, delimiter=",", **kwargs) -> Tuple[bool, Any]:
        nstr = []
        for saved_entry in self.memory:
            nstr.append(
                delimiter.join(saved_entry) + "\n"
            )
        return (True, "".join(nstr))


    def _serialize(self, **kwargs) -> Tuple[bool, List[Tuple[str, str, str, int, int, float]]]:
        return self.memory


    def _clear_state(self, **kwargs):
        pass

    # TODO need to implement serialization code as well...


    

