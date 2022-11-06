"""
Module containing base classes for EdnaML metrics.
"""

try:
    import torchmetrics
except ImportError:
    print('WARNING: torchmetrics is not installed!')
    pass

class BaseMetric:
    """Base class for EdnaML metrics."""
    def __init__(self, metric_name, metric_type):
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.build_module()
        #self.metric_params =  self.build_params() holding off
        self.post_init_val()

    def build_module(self, **kwargs):
        """Set up necessary object computation parameters."""
        raise NotImplementedError

    def build_params(self):
        """Return metric parameters for the BaseMetric."""
        raise NotImplementedError

    def post_init_val(self):
        """Perform post-initialization validation."""
        raise NotImplementedError

    def update(self, **kwargs):
        """Perform computation step using args"""
        raise NotImplementedError

    def clear(self, **kwargs):
        """Clear/reset object computation memory."""
        for param in self.metric_params:
            self.metric_params[param] = []

    def save(self, **kwargs):
        """Perform a save routine - either local or remote."""
        raise NotImplementedError

    def backup(self, **kwargs):
        """Perform backup/sync to cloud."""
        raise NotImplementedError

    def _LocalMetricSave(self, **kwargs):
        """Save computation results locally."""
        raise NotImplementedError

    def _RemoteMetricSave(self, **kwargs):
        """Save computation results to remote location specified in config YAML."""
        raise NotImplementedError

    def print_info(self):
        print(f'INFO for metric {self.metric_name}:')
        print(f'Metric Type: {self.metric_type}')
        print(f'Parameters: {self.metric_params}')