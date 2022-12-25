import logging
from ednaml.metrics.BaseMetric import BaseMetric
from ednaml.metrics.BaseTorchMetric import BaseTorchMetric
from ednaml.config.MetricsConfig import MetricsConfig
from ednaml.metrics.AdHocMetric import AdHocMetric
from typing import Dict, Type, List, Any
from ednaml.utils import ERSKey, StorageArtifactType, locate_class
from ednaml.storage import BaseStorage
from ednaml.storage import StorageManager
from ednaml.utils.LabelMetadata import LabelMetadata

class MetricsManager:
    # All metrics that have been created. 
    # Later we will replace with model/log, etc specifics...
    metrics: Dict[str, BaseMetric]
    log_metrics: List[BaseMetric]
    model_metrics: List[BaseMetric]
    artifact_metrics: List[BaseMetric]
    plugin_metrics: List[BaseMetric]
    code_metrics: List[BaseMetric]
    config_metrics: List[BaseMetric]
    metric_metrics: List[BaseMetric]

    params: Dict[str, Any]
    metrics_enable: bool
    always_enable: bool
    step_enable: bool
    batch_enable: bool
    metrics: Dict[str, BaseMetric]
    always_metrics: List[str]
    immediate_metrics: List[str]
    step_metrics: List[str]
    batch_metrics: List[str]

    will_save_itself: List[str]
    will_need_storage: Dict[str, str]   # Dict of metrics to storage_names
    will_need_serializing: List[str]
    storage_groups: Dict[str, List[str]] # Dict of storage to list of metrics

    storage_manager: StorageManager
    storage: Dict[str, BaseStorage]


    def __init__(self, metrics_config: MetricsConfig, logger: logging.Logger, storage: Dict[str, BaseStorage], skip_metrics: List[str] = []):
        # Called in EdnaML's apply()--> buildMetricsManager()
        self.storage = storage
        self.metrics = {}
        self.params = {}
        self.logger = logger
        self.will_save_itself = []              # metrics that save themselves
        self.will_need_storage = {}             # metrics that save to a defined Storage
        self.will_need_serializing = []         # Metrics saved to file (or serialized (TODO alter))
        self.local_file = "metrics.json"
        # TODO update this to not JUST model metrics...a
        self.model_metrics = []
        self.log_metrics, self.model_metrics, self.artifact_metrics, self.plugin_metrics, self.config_metrics, self.code_metrics, self.metric_metrics = [],[],[],[],[],[],[]
        metric_mappings = {"log": (metrics_config.LOG_METRICS, self.log_metrics),
                                                            "model": (metrics_config.MODEL_METRICS, self.model_metrics),
                                                            "artifact": (metrics_config.ARTIFACT_METRICS, self.artifact_metrics),
                                                            "plugin": (metrics_config.PLUGIN_METRICS, self.plugin_metrics),
                                                            "config": (metrics_config.CONFIG_METRICS, self.config_metrics),
                                                            "code": (metrics_config.CODE_METRICS, self.code_metrics),
                                                            "metric": (metrics_config.METRIC_METRICS, self.metric_metrics),
                                                            }
        for metric_type in metric_mappings:
            if metric_type in skip_metrics:
                self.log("Skipping metric type {metric_type} due to `skip_metrics`".format(metric_type = metric_type))
                continue
            

            for metric_option_config in metric_mappings[metric_type][0]: # Iterate over all metrics

                # Identify the metric class
                metric_class: Type[BaseMetric] = locate_class(
                    subpackage = "metrics",
                    classpackage=metric_option_config.METRIC_CLASS, # "TorchMetricAccuracy"
                )
                # Instantiate the metric
                self.metrics[metric_option_config.METRIC_NAME] = metric_class(
                    metric_name = metric_option_config.METRIC_NAME,
                    metric_type = StorageArtifactType(metric_type), # e.g. model | log | config, etc
                    metric_trigger = metric_option_config.METRIC_TRIGGER,
                )
                #Apply the arguments
                self.metrics[metric_option_config.METRIC_NAME].apply(metric_option_config.METRIC_ARGS, metric_option_config.METRIC_PARAMS)
                self.log("Added metric %s with class %s"%(metric_option_config.METRIC_NAME, metric_option_config.METRIC_CLASS))
                if self.metrics[metric_option_config.METRIC_NAME].will_save_itself:
                    self.will_save_itself.append(metric_option_config.METRIC_NAME)
                else:
                    if metric_option_config.METRIC_STORAGE is not None:
                        self.will_need_storage[metric_option_config.METRIC_NAME] = metric_option_config.METRIC_STORAGE
                    else:
                        self.will_need_serializing.append(metric_option_config.METRIC_NAME)
                
                # Add a reference to the metric into the respective list
                metric_mappings[metric_type][1].append(self.metrics[metric_option_config.METRIC_NAME])

        self.storage_groups = {}
        for metric_name in self.will_need_storage:
            if self.will_need_storage[metric_name] not in self.storage_groups:
                self.storage_groups[self.will_need_storage[metric_name]] = []
            self.storage_groups[self.will_need_storage[metric_name]].append(metric_name)


        self.ad_hoc_metric = AdHocMetric(metric_name="reserved-adhoc-metric", metric_type="reserved-type", metric_trigger = "other")
        self.ad_hoc_metric.apply(metric_kwargs={}, metric_params={
            "metric_name":"metric_name",
            "metric_val":"metric_val",
            "metric_type":"metric_type"
        })


        self.metrics_enable = False
        self.always_enable = False
        self.step_enable = False
        self.batch_enable = False

        self.always_metrics = []
        self.immediate_metrics = []
        self.step_metrics = []
        self.batch_metrics = []

        self.internal_metrics = {}

    def log(self, msg):
        self.logger.info("[metrics-manager]" + msg)

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
    def log_metric(self, metric_name, epoch, step, value, metric_type = "model"):
        self.ad_hoc_metric.update(epoch=epoch, step=step,params={"metric_name":metric_name, "metric_type": metric_type, "metric_val": value, "metric_class":"adhoc"})


    def getLogMetrics(self) -> List[BaseMetric]:
        return self.log_metrics
    def getArtifactMetrics(self) -> List[BaseMetric]:
        return self.artifact_metrics
    def getModelMetrics(self) -> List[BaseMetric]:
        return self.model_metrics
    def getPluginMetrics(self) -> List[BaseMetric]:
        return self.plugin_metrics
    def getConfigMetrics(self) -> List[BaseMetric]:
        return self.config_metrics
    def getCodeMetrics(self) -> List[BaseMetric]:
        return self.code_metrics
    def getMetricMetrics(self) -> List[BaseMetric]:
        return self.metric_metrics

    def addMetrics(self, metrics_list: List[BaseMetric], epoch, step):
        self.immediate_metrics: List[BaseMetric] = []
        if len(metrics_list) > 0:
            # Check for metrics that are executed only once, e.g. now
            for metric in metrics_list:
                self.internal_metrics[metric.metric_name] = metric
                if metric.metric_trigger == "once":
                    # Metric is triggered only once.
                    # We will trigger it at the end of addMetrics()

                    self.immediate_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "always":
                    # Metric is `always` triggered. Always has different meanings for each Manager. We will add it to our bookkeeping
                    self.always_metrics.append(metric.metric_name)

                elif metric.metric_trigger == "step":
                    self.step_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "batch":
                    self.batch_metrics.append(metric.metric_name)
                else:
                    self.logger.info("metric_trigger %s is not supported. Assuming this is AdHoc metric."%metric.metric_trigger)
            self.metrics_enable = True
        else:
            self.metrics_enable = False

        if self.metrics_enable:
            for metric_name in self.immediate_metrics:
                self.internal_metrics[metric_name].update(epoch, step, params=self.params)

            if len(self.always_metrics):
                self.always_enable = True # Enable always metrics
            if len(self.step_metrics):
                self.step_enable = True # Enable step metrics
            if len(self.batch_metrics):
                self.batch_enable = True # Enable batch metrics

    # For now, we ignore always metrics
    def updateStepMetrics(self, epoch, step):
        for metric_name in self.step_metrics:
            self.internal_metrics[metric_name].update(epoch=epoch, step=step, params=self.params)

    def updateBatchMetrics(self, epoch, step):
        for metric_name in self.batch_metrics:
            self.internal_metrics[metric_name].update(epoch=epoch, step=step, params=self.params)

    def flush(self):
        pass



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


    def save(self,epoch, step):
        with open(self.local_file, "a") as lfile:
            for metric_name in self.will_need_serializing:
                success, serialized_metrics = self.metrics[metric_name].serializer()
                lfile.write(serialized_metrics) # serialized_metrics format is a string, each line is a csv object
                self.metrics[metric_name].clear()
            success, serialized_metrics = self.ad_hoc_metric.serializer()
            if len(serialized_metrics):
                lfile.write(serialized_metrics) # serialized_metrics format is a string, each line is a csv object

        for storage_name in self.storage_groups:
            storage_file = "_".join([storage_name, "metrics"])+".json"
            with open(storage_file, "a") as lfile:
                for metric_name in self.storage_groups[storage_name]:
                    success, serialized_metrics = self.metrics[metric_name].serializer()
                    lfile.write(serialized_metrics)
                self.metrics[metric_name].clear()
            
            upload_ers_key = self.storage_manager.getERSKey(epoch=epoch, step=step, artifact_type=StorageArtifactType.METRIC)
            self.storage[storage_name].upload(
                ers_key=upload_ers_key,
                source_file_name=storage_file,
                canonical=False,    # TODO deal with this...
            )

        for metric_name in self.will_save_itself:
            self.metrics[metric_name].save()
            self.metrics[metric_name].clear()

        # TODO SaveRecords???

    def updateMetadata(self, label_metadata: LabelMetadata):
        for metric_name in self.metrics:
            self.metrics[metric_name].updateMetadata(label_metadata=label_metadata)

        
    def clear(self):
        for metric_name in self.metrics:
            self.metrics[metric_name].clear()


    def getLocalFile(self) -> str:
        return self.local_file


    def updateERSKey(self, ers_key: ERSKey, file_name: str, storage_manager: StorageManager):
        """Update the logger with the run information from the ERSKey. The StorageKey is also available
        if this logger indexes logs as such.

        The file_name is the local file where logs can be dumped. A Log Storage will upload this local file
        to its remote Storage with the latest ERS-Key. This means a batch of logs is generally indexed by the
        logging backup frequency (depending on how the Log Storage takes care of files)

        Args:
            ers_key (ERSKey): _description_
            file_name (str): _description_
        """
        self._registerStorageManager(storage_manager)
        self._registerSaveInformation(ers_key, file_name)
        
        
    def _registerStorageManager(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager

    def _registerSaveInformation(self, ers_key: ERSKey, file_name: str):
        """We update the local file with the provided file.

        Args:
            ers_key (ERSKey): _description_
            file_name (str): _description_
        """
        self.local_file = file_name
        if file_name == "":
            self.logger.info("Not writing metrics due to Empty storage.") # TODO
        else:
            self.logger.info("Will write serializable metrics to local file {fname}".format(fname = file_name))


