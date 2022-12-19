"""
Module containing base classes for EdnaML metrics.
"""

import json
from abc import ABC, abstractmethod
"""

What is stored in a local serialized metric:

epoch, step, metric_value

what is saved in serialized metric:

(metric_name, metric_type, metric_class, epoch, step, metric_value)


What is stored in a fully specified metric:

experiment_key, run_key, epoch, step, metric_name, metric_type, metric_class, metric_value



"""



class BaseMetric(ABC):
    """Base class for EdnaML metrics."""
    will_save_itself: bool
    # Instantiate the metric class
    def __init__(self, metric_name, metric_type):
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.state = {}
        self.memory = []    # list of (epoch, step, value)
        self.will_save_itself = False
        self.params_dict = {}
        self.clear()
    

    # Build the metric itself
    def apply(self, metric_kwargs, metric_params):
        self.build_metric(**metric_kwargs)
        self.add_params(metric_params)  
        self.post_init_val(**metric_kwargs)

    @abstractmethod
    def build_metric(self, **kwargs):
        """Set up necessary object computation parameters."""
        raise NotImplementedError

    
    def add_params(self, metric_params):
        self.params_dict = self._add_params(metric_params)

    @abstractmethod
    def _add_params(self, metric_params):
        return metric_params

    @abstractmethod
    def post_init_val(self, **kwargs):
        """Perform post-initialization validation."""
        raise NotImplementedError

    
    def update(self, epoch, step, params):
        """Compute metric, and save to memory"""
        params_dict = {key:params[self.params_dict[key]] for key in self.params_dict}
        response = self._compute_metric(epoch, step, params_dict)
        success = self._add_value(epoch, step, response)
        return success

    @abstractmethod
    def _compute_metric(self, epoch, step, **kwargs) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _add_value(self, epoch, step, response) -> bool:
        self.memory.append((epoch, step, response))
        return True

    @abstractmethod
    def clear(self, **kwargs):
        """Clear/reset object computation memory."""
        self.state = {}
        self.memory = []

    @abstractmethod
    def save(self, **kwargs):
        """Perform a save routine - either local or remote."""
        print(self.state)

    @abstractmethod
    def backup(self, **kwargs):
        """Perform backup/sync to cloud."""
        raise NotImplementedError

    @abstractmethod
    def _LocalMetricSave(self, **kwargs):
        """Save computation results locally."""
        raise NotImplementedError

    @abstractmethod
    def _RemoteMetricSave(self, **kwargs):
        """Save computation results to remote location specified in config YAML."""
        raise NotImplementedError

    
    def print_info(self):
        print(f'INFO for metric {self.metric_name}:')
        print(f'Metric Type: {self.metric_type}')
        try:
            print(f'Parameters: {self.metric_params}')
        except NameError:
            print('WARNING: build_module has not set up metric params.')
    def print_state(self):
        print(f'STATE of metric {self.metric_name}:')
        print(json.dumps(self.state,indent=2))