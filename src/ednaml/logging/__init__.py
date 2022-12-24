import logging
from typing import List
from ednaml import storage
from ednaml.metrics.BaseMetric import BaseMetric
from ednaml.storage import StorageManager
from ednaml.utils import ERSKey, ExperimentKey, StorageArtifactType
from typing import Dict, Any

class LogManager:
    """LogManager is the base class for logging in EdnaML.

    TODO: Update all loggers to use LogManager, not python logger, in their calls

    Parameters:
        experiment_name
        has_logger
        [crit|error|warn|info|debug]_logs: Number of [crit|error|...] messages logged since last flush.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    experiment_key: ExperimentKey
    metrics_enable: property
    params: Dict[str, Any]
    storage_manager: StorageManager
    metrics: Dict[str, BaseMetric]
    always_metrics: List[str]
    immediate_metrics: List[str]
    step_metrics: List[str]
    batch_metrics: List[str]
    logger: logging.Logger
    log_levels = {
        "critical" : 50,
        "fatal" : 50,
        "error" : 40,
        "warning" : 30,
        "warn" : 30,
        "info" : 20,
        "debug" : 10,
        "notset" : 0,
    }


    def __init__(self, experiment_key: ExperimentKey, **kwargs):
        self.params = {}
        self.experiment_key = experiment_key
        self.logger = None
        
        self.metrics_enable = False
        self.always_enable = False
        self.step_enable = False
        self.batch_enable = False

        self.always_metrics = []
        self.immediate_metrics = []
        self.step_metrics = []
        self.batch_metrics = []

        self.metrics = {}
        

        self.params["experiment_name"] = self.experiment_key.getExperimentName()
        self.params["has_logger"] = False
        self.apply(**kwargs)

    @property
    def metrics_enable(self):
        return self.params.get("metrics_enable", False)
    @metrics_enable.setter
    def metrics_enable(self, val: bool):
        self.params["metrics_enable"] = val


    @property
    def crit_logs(self):
        return self.params.get("crit_logs", 0)
    @crit_logs.setter
    def crit_logs(self, val: bool):
        self.params["crit_logs"] = val
    
    @property
    def error_logs(self):
        return self.params.get("error_logs", 0)
    @error_logs.setter
    def error_logs(self, val: bool):
        self.params["error_logs"] = val
    
    @property
    def info_logs(self):
        return self.params.get("info_logs", 0)
    @info_logs.setter
    def info_logs(self, val: bool):
        self.params["info_logs"] = val
    
    @property
    def debug_logs(self):
        return self.params.get("debug_logs", 0)
    @debug_logs.setter
    def debug_logs(self, val: bool):
        self.params["debug_logs"] = val
    
    @property
    def warn_logs(self):
        return self.params.get("warn_logs", 0)
    @warn_logs.setter
    def warn_logs(self, val: bool):
        self.params["warn_logs"] = val




    def apply(self, **kwargs):
        """Build the logger internal state. At this time, the logger does not have access
        to the ERSKey, or any indexing information about the current experiment.

        This function can be used to initialize logging, and set of batched requests once
        the logger has access to indexing information.
        """
        raise NotImplementedError()
    
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
        raise NotImplementedError()

    def getLogger(self):
        return self

    def flush(self) -> bool:
        """Flush any remaining logs.

        Returns:
            bool: Flush success
        """
        pass


    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level=self.log_levels[level], msg = msg, *args)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg=msg, *args, **kwargs)
        self.crit_logs+=1
        # TODO for more fancy stuff in future, we can add more bookkeeping, e.g. if params_enable["crit_logs"]: crit_logs += 1 ; for metrics_needing["crit_logs"] -> if metric_trigger_reached: metric.update(epoch, step, etc)
    def fatal(self, msg, *args, **kwargs):
        self.logger.fatal(msg=msg, *args, **kwargs)
        self.crit_logs+=1
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg=msg, *args, **kwargs)
        self.error_logs+=1
    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg=msg, *args, **kwargs)
        self.warn_logs+=1
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg=msg, *args, **kwargs)    
        self.warn_logs+=1
    def info(self, msg, *args, **kwargs):
        self.logger.critical(msg=msg, *args, **kwargs)
        self.info_logs+=1
    def debug(self, msg, *args, **kwargs):
        self.logger.critical(msg=msg, *args, **kwargs)
        self.debug_logs+=1
    

    def getLocalLog(self) -> str:
        """Return path to a local file containing any disk logs. Can be an empty file.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: Path to log file
        """
        raise NotImplementedError()

    def addMetrics(self, metrics_list: List[BaseMetric], epoch, step):
        self.immediate_metrics: List[BaseMetric] = []
        if len(metrics_list) > 0:
            # Check for metrics that are executed only once, e.g. now
            for metric in metrics_list:
                self.metrics[metric.metric_name] = metric
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
                    raise ValueError("metric_trigger %s is not supported"%metric.metric_trigger)
            self.metrics_enable = True
        else:
            self.metrics_enable = False

        if self.metrics_enable:
            for metric_name in self.immediate_metrics:
                self.metrics[metric_name].update(epoch, step, params=self.params)

            if len(self.always_metrics):
                self.always_enable = True # Enable always metrics
            if len(self.step_metrics):
                self.step_enable = True # Enable step metrics
            if len(self.batch_metrics):
                self.batch_enable = True # Enable batch metrics

    # For now, we ignore always metrics
    def updateStepMetrics(self, epoch, step):
        for metric_name in self.step_metrics:
            self.metrics[metric_name].update(epoch=epoch, step=step, params=self.params)

    def updateBatchMetrics(self, epoch, step):
        for metric_name in self.batch_metrics:
            self.metrics[metric_name].update(epoch=epoch, step=step, params=self.params)


    # Metricsmanager takes care of saving


# So, here's the idea...
# Each metric has certain params it requires
# Each manager has certain params it publishes
# We do not need to publish all params if metrics only require a subset (DEAL WITH THIS LATER, TODO)
# How are params published...?
# Each time we do a task, we publish a relevant param, e.g. update a params dict inside our manager
# But what if param is not needed...?
# How do we seamlessly NOT update a param?