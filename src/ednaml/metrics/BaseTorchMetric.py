from ednaml.metrics.BaseMetric import BaseMetric
import json, torch
from ednaml.utils import locate_class
from typing import Type, Any, Tuple, List, Dict


class BaseTorchMetric(BaseMetric):
    """Wrapper class for TorchMetrics metrics for use in EdnaML."""
    
    def build_metric(self, **kwargs):
        self.metric_class = kwargs.get("metric_class")
        self.aggregate = kwargs.get("aggregate", False)
        metric_kwargs = kwargs.get("metric_kwargs")
        from torchmetrics import Metric
        metric_class: Type[Metric] = locate_class(package="torchmetrics", subpackage=self.metric_class)
        self._metric = metric_class(**metric_kwargs)
        self.metric_class = "TorchMetric"+self.metric_class

    def post_init_val(self, **kwargs):
        pass # TODO???
        return (True, "")


    def update(self, epoch: int, step: int, params: Dict[str, Any]) -> bool:
        required_params = self._get_required_params(params=params)
        response = self._compute_metric(epoch, step, **required_params)
        if not self.aggregate:
            success = self._add_value(epoch, step, self._getPrimitive(self._metric.compute()))
        else:
            success = True
            if step%self.aggregate == 0: 
                success = self._add_value(epoch, step, self._getPrimitive(self._metric.compute()))
                self._metric.reset()
        return success

    def _getPrimitive(self, data: torch.Tensor):
        """Converts the torch tensor to primitive. TODO update this for ConfusionMatrix and other variants that return an array...

        Args:
            data (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return data.item()

    def _compute_metric(self, epoch: int, step: int, **kwargs) -> float:
        return self._metric(**kwargs)

    def getLastVal(self):
        return "%.3f"%self.last_val


    # TODO: finish the metric compute
    # Also need to handle logging of params
    # for exampe, if I just want to log the loss, or a raw param from a Manager, how do I do that?
    # AdHocMetric...?
    # Yep -- add a METRICS thing to TRAINER (and others?) to directly log params
    # so, in BaseTrainer, we can directly to self.log_metric("loss", lossbackward) instead of creating a Metric specifically to track it.
    # log_metric will go under adhoc metric. We might want to rename it from adhoc to something else???



    # def save(self, epoch, batch, result):
    #     # Ensure the result is JSON-serializable
    #     try:
    #         json.dumps(result)
    #     except TypeError:
    #         raise ValueError(f'The computed result of the EdnaML_TorchMetric \' {self.metric_name} \' ({result}) is not JSON-serializable.')
    #     # Key=Type-Metric-Epoch-Batch, Value=Result
    #     self.state[self.metric_type][self.metric_name][epoch] = {batch: result}

    # [(metric_name, metric_type, metric_class, epoch, step, metric_value)]
    def _serialize_to_string(self, delimiter=",", **kwargs) -> Tuple[bool, Any]:
        nstr = []
        for saved_entry in self.memory:
            nstr.append(
                delimiter.join([self.metric_name, self.metric_type.value, self.metric_class, str(saved_entry[0]), str(saved_entry[1]), str(saved_entry[2])]) + "\n"
            )
        return (True, "".join(nstr))


    def _serialize(self, **kwargs) -> Tuple[bool, List[Tuple[str, str, str, int, int, float]]]:
        return (True, [(self.metric_name, self.metric_type.value, self.metric_class, saved_entry[0], saved_entry[1], saved_entry[2]) for saved_entry in self.memory])


    def _clear_state(self, **kwargs):
        self._metric.reset()