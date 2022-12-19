from ednaml.metrics.BaseMetric import BaseMetric
from ednaml.metrics.BaseTorchMetric import BaseTorchMetric
from ednaml.config.MetricsConfig import MetricsConfig
from ednaml.metrics.AdHocMetric import AdHocMetric
from typing import Dict, Type, List, Any
from ednaml.utils import locate_class

class MetricsManager:
    # All metrics that have been created. 
    # Later we will replace with model/log, etc specifics...
    metrics: Dict[str, BaseMetric]  
    def __init__(self, metrics_config: MetricsConfig):
        # Called in EdnaML's apply()--> buildMetricsManager()
        self.metrics = {}
        self.will_save_itself = []
        self.will_need_storage = {}
        self.will_need_serializing = []
        # TODO update this to not JUST model metrics...a
        self.model_metrics = []
        for metric_option_config in metrics_config.MODEL_METRICS: # Iterate over all metrics
            # Identify the metric class
            metric_class: Type[BaseMetric] = locate_class(
                subpackage = "metrics",
                classpackage=metric_option_config.METRIC_CLASS, # "TorchMetricAccuracy"
                classfile = "model"
            )
            # Instantiate the metric
            self.metrics[metric_option_config.METRIC_NAME] = metric_class(
                metric_name = metric_option_config.METRIC_NAME,
                metric_type = metric_option_config.METRIC_TYPE,
            )
            #APply the arguments
            self.metrics[metric_option_config.METRIC_NAME].apply(metric_option_config.METRIC_ARGS, metric_option_config.METRIC_PARAMS)

            if self.metrics[metric_option_config.METRIC_NAME].will_save_itself:
                self.will_save_itself.append(metric_option_config.METRIC_NAME)
            else:
                if metric_option_config.METRIC_STORAGE is not None:
                    self.will_need_storage[metric_option_config.METRIC_NAME: metric_option_config.METRIC_STORAGE]
                else:
                    self.will_need_serializing.append(metric_option_config.METRIC_NAME)
            self.model_metrics.append(metric_option_config.METRIC_NAME)

        self.ad_hoc_metric = AdHocMetric(metric_name="reserved-adhoc-metric", metric_type="reserved-type")
        self.ad_hoc_metric.apply(metric_kwargs={}, metric_params={
            "metric_name":"metric_name",
            "metric_val":"metric_val",
            "metric_type":"metric_type"
        })

    def updateMetric(self, metric_name, epoch, step, params):
        self.metrics[metric_name].update(epoch=epoch, step = step, params=params)

    # create updateLogMetrics, etc, etc.
    # These are called inside respective managers. e.g. LogManager, when it gets a log message, will call log-metrics
    # We can improve efficiency by having a bookkeeping agent inside the manager
    # This bookkeeping agent works with metric manager to figure out what params are being tracked
    # Then, it calls metric update only if those params exist in scope.
    # example: if we have a LogMetric that tracks numLogsSaved, then we do not need to call updateLogMetric in the logging step (we will need to call it in the log saving step, tho)
    # on the other hand, if we have a LogMetric that tracks logLength, then we DO need to call updateLogMetric in the logging step
    def updateModelMetrics(self, epoch, step, params):
        for metric_name in self.model_metrics:
            # NOTE: nonblocking metrics, i.e. ones that take care of themselves, use futures, and crate copies of passed variables
            self.updateMetric(metric_name, epoch, step, params)

    # Now, this is a tricky thing
    # In general, for a structured pipeline, metrics SHOULD take care of themselves
    # But sometimes, as we tool out, we won't have the metrics classes built properly
    # So we might track some metric ad hoc until we built the tooling
    # Ex. we are doing text classification, and want to tracking attention-l2 of final attention layer
    # So, in our custom trainer, without dealing with all the brouhaha, we can manually compute the attention-l2, then
    # self.log_metric("attention_l2", val)
    # In turn, log_metric in trainer calls MM.logMetric(attention_l2, self.global_epocj, self.global_batch, val, LOG)
    def logMetric(self, metric_name, epoch, step, value, metric_type):
        


    

    # MetricsManager can save metrics in 3 ways:
    # serialize all metrics together into a single file
    #   this is the 'simple' case
    #   Here, we follow the logmanager approach
    #   in trainer, we use the saving mechanism (i.e. call the serialize, write to file, transfer to local storage, then upload)
    #   in other modes, calling the serializer SHOULD generate an empty file with nothing to upload
    # each metric manages its own saving. 
    #   In these cases, a developer might choose to use a bespoke set of metrics packaged with each experiment
    #   For metrics managing its own saves, they should take care of it whether asynchronous or synchronously
    #   for now we will create synchronous
    #   later, we will adopt the serving approach to make them asynchronous (just like model serving)
    #   We can create antoerh top-level framework inherited from BaseTrainer for ease of use, but basically, it's all in the metrics itself
    # each metric is tied to an individual storage, to collect different types of metrics. THis is managed in MetricsManager section of config
    #   This is fine-grained and coarse. We can individually determine each metric's storage, and give categories, likle logmetrics, their own storage for every logmetrics...
    #   The way this would work, in practice, is in config.METRICS_MANAGER, for each 


    # METRICS:
    #  - METRIC_NAME
    #    METRIC_CLASS
    #    METRIC_PARAMS
    #    METRIC_ARGS
    #    METRIC_STORAGE: null | something
    
    # So, if METRIC_STORAGE is null, then METRIC either manages its own saving or needs help
    #       Each BaseMetric contains a `will_save_itself` boolean. MetricsManager will, when instantiating metrics, track which metrics `will_save_itself`
    #       If this list of NOT(`will_save_itself`) is empty, means all metrics are either managing their own saving or have storage
    # If Metric_Storage contains a storage item, then, when MetricsManager is asked to serialize the metrics it contains
    #       for metrics that manage themselves, they return nothing
    #       for metrics that dont manage, they add their serialized list to the global metrics save
    #       we will probably eventually decide to split this into individual files, but for now, eh
    #       for metrics tied to a storage, we directly provide the serialization to the storage, without saving
    #       This presents a small complication and chance for improvement
    #       we need to add an option for a storage -- does it require a file, a serialized string, or serialized bytes?
    #       Then, we can call the appropriate serializer for model, logs, config, etc: serialize(), save(), bSerialize(), streamSerialize, bStreamSerialize


    def 

