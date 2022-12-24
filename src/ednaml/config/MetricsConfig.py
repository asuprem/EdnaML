


from typing import Any, Dict, List, Union
from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults

class MetricOptionsConfig(BaseConfig):
    METRIC_NAME: str
    METRIC_CLASS: str
    METRIC_PARAMS: Dict[str, str]
    METRIC_ARGS: Dict[str, str]
    METRIC_STORAGE: Union[str,None]
    METRIC_TRIGGER: str

    def __init__(self, metric_options_dict):
        self.METRIC_NAME = metric_options_dict.get("METRIC_NAME")                   # A unique name for this metric for referencing
        self.METRIC_CLASS = metric_options_dict.get("METRIC_CLASS")                 # The class for this Metric to instantiate it
        self.METRIC_PARAMS = metric_options_dict.get("METRIC_PARAMS", {})           # the param mapping for this metric
        self.METRIC_ARGS = metric_options_dict.get("METRIC_ARGS", {})               # the arguments to instantiate this metric
        self.METRIC_STORAGE = metric_options_dict.get("METRIC_STORAGE", None)       # the storage this metric tracks, if needed
        self.METRIC_TRIGGER = metric_options_dict.get("METRIC_TRIGGER", "batch")   # How often to trigger: `once`, `always`, `step`, `batch`


        # Re. Metric_Trigger:
        #       once: trigger after metrics provided to manager, once
        #       always: trigger at each params update step. Expensive and bad, usually
        #       step: trigger at each step
        #       always: trigger at each X steps, X from LOGGING.STEP_VERBBOSE


class MetricsConfig(BaseConfig):
    LOG_METRICS: List[MetricOptionsConfig]
    MODEL_METRICS: List[MetricOptionsConfig]
    ARTIFACT_METRICS: List[MetricOptionsConfig]
    CONFIG_METRICS: List[MetricOptionsConfig]
    PLUGIN_METRICS: List[MetricOptionsConfig]
    METRIC_METRICS: List[MetricOptionsConfig]
    CODE_METRICS: List[MetricOptionsConfig]
    EXTRA_METRICS: List[MetricOptionsConfig]

    def __init__(self, metric_dict, defaults: ConfigDefaults):
        self.LOG_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("LOG_METRICS", [])]
        self.MODEL_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("MODEL_METRICS", [])]
        self.ARTIFACT_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("ARTIFACT_METRICS", [])]
        self.CONFIG_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("CONFIG_METRICS", [])]
        self.PLUGIN_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("PLUGIN_METRICS", [])]
        self.METRIC_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("METRIC_METRICS", [])]
        self.CODE_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("CODE_METRICS", [])]
        self.EXTRA_METRICS = [MetricOptionsConfig(item) for item in metric_dict.get("EXTRA_METRICS", [])]
        
        