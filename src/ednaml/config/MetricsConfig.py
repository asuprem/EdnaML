


from typing import Any, Dict, List
from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults

class MetricOptionsConfig(BaseConfig):
    METRIC_NAME: str
    METRIC_CLASS: str
    METRIC_TYPE: str
    METRIC_PARAMS: Dict[str, str]
    METRIC_ARGS: Dict[str, str]
    METRIC_STORAGE: str

    def __init__(self, metric_options_dict):
        self.METRIC_NAME = metric_options_dict.get("METRIC_NAME")
        self.METRIC_CLASS = metric_options_dict.get("METRIC_CLASS")
        self.METRIC_TYPE = metric_options_dict.get("METRIC_TYPE")
        self.METRIC_PARAMS = metric_options_dict.get("METRIC_PARAMS", {})
        self.METRIC_ARGS = metric_options_dict.get("METRIC_ARGS", {})
        self.METRIC_STORAGE = metric_options_dict.get("METRIC_STORAGE", None)


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
        
        