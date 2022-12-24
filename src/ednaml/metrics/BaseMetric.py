"""
Module containing base classes for EdnaML metrics.
"""

from configparser import NoOptionError
import json
from abc import ABC, abstractmethod
from types import NoneType
from typing import List, Dict, Any, Tuple
from ednaml.utils import StorageArtifactType
from ednaml.storage import BaseStorage
from ednaml.utils.LabelMetadata import LabelMetadata
"""

What is stored in a local serialized metric:

epoch, step, metric_value

what is saved in serialized metric:
<name>         <log/etc>    <class name for metric, e.g. adhoc, Accuracy, TorchAccuracy, etc>
(metric_name, metric_type, metric_class, epoch, step, metric_value)


What is stored in a fully specified metric:

experiment_key, run_key, epoch, step, metric_name, metric_type, metric_class, metric_value



"""



class BaseMetric(ABC):
    """Base class for EdnaML metrics."""
    will_save_itself: bool              # Whether Metric manages it's own saving, serialization, and logging through args.
    metric_name: str                    # Name for metric (unique across all metrics in workspace)
    metric_type: StorageArtifactType    # Type for metric, e.g. MODEL, CONFIG, LOG, etc...
    state: Dict[str, Any]               # The state parameters for this metric, for use in saving metric state. Unused for now.
    memory: List[Tuple(int, int, float)]                 # Temporary memory for recent metric computations, for batching metric uploads. Useful if not saving itself.
    params_dict: Dict[str,str]          # Dictionary of key-value pairs. Key are arguments for _compute_metric, and values are corresponding keys in respective params
    metric_storage: BaseStorage         # If provided, metric will save to this storage
    to_serialize: bool                  # Whether to serialize into a list of tuples [()], or save to file
    trigger: str                        # How often to trigger the metric, from METRIC_TRIGER. `once`, `always`

    # Note: serialization format:   [(metric_name, metric_type, metric_class, epoch, step, metric_value)]. This is combined with ers-key in the storage.

    def __init__(self, metric_name: str, metric_type: StorageArtifactType, metric_trigger: str):
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.state = {}     # temporary metric state to keep for a batch. TODO later.
        self.memory = []    # list of (epoch, step, value)
        self.will_save_itself = False
        self.params_dict = {}
        self.metric_trigger = metric_trigger
        self.metric_class = self.__class__.__name__
        self.clear()

        # This is for forward compatibility for when we enable fine-grained storage
        self.metric_storage = None
        if self.metric_storage is not None:
            self.to_serialize = False   # TODO use storage to determine if this should be serialized or not...
        # to_serialize = True --> return string ; to_serialize = False --> return Tuple
        self.to_serialize = True


        if self.will_save_itself:
            self.save = self._save
            self.serializer = self._noop_serialize
        else:
            if self.metric_storage is not None:
                # Note: _storage_save may use to_serialize value to determine if metrics should be serialized or not
                self.save = self._storage_save
            else:
                self.save = self._metric_save
        
            if self.to_serialize:
                self.serializer = self._serialize_to_string
            else:
                self.serializer = self._serialize


    def updateMetadata(self, label_metadata: LabelMetadata):
        pass

    # Build the metric itself
    def apply(self, metric_kwargs: Dict[str, Any], metric_params: Dict[str, str]):
        self.build_metric(**metric_kwargs)
        self.add_params(metric_params)  
        (success, msg)= self.post_init_val(**metric_kwargs)
        if not success: # TODO add option for ignore_if_error, and disable metric.
            raise ValueError("Failed to validate metric `{metric_name}`, with error: \t{msg}".format(metric_name=self.metric_name, msg = msg))

    @abstractmethod
    def build_metric(self, **kwargs):
        """Build the metric object using kwargs. Metric-specific kwargs can be fully specified in method signature, as long as kwargs is also provided. 

        Provided kwargs are METRIC_KWARGS from configuration.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    
    def add_params(self, metric_params: Dict[str, str]):
        """Add the provided `params` mapping to the Metric's `params_dict`

        Note on params_dict
        For example, an Accuracy metric usually takes the arguments 
        `preds` and `target`. However, if using Accuracy with HFTrainer, 
        which provides metrics with `labels`, and not `targets`, we can use 
        `params_dict` to map `labels` to `targets` so our Accuracy metric
        knows to use `labels` as a proxy for `targets`

        Args:
            metric_params (Dict[str, str]): Mapping of _compute_metric params to EdnaML object's params
        """
        self.params_dict = self._add_params(metric_params)

    @abstractmethod
    def _add_params(self, metric_params: Dict[str, str]) -> Dict[str, str]:
        """Internal method to add params. Override for custom _add_params.

        Args:
            metric_params (Dict[str, str]): Mapping of _compute_metric params to EdnaML object's params

        Returns:
            Dict[str, str]: Mapping of _compute_metric params to EdnaML object's params, with any desired modifications.
        """
        return metric_params

    @abstractmethod
    def post_init_val(self, **kwargs)->Tuple[bool, str]:
        """Perform post-initialization validation. 

        Returns:
            Tuple[bool, str]: A tuple of initialization success boolean and error message.
        """
        return (False, "NotImplementedError")

    
    def update(self, epoch: int, step: int, params: Dict[str, Any]) -> bool:
        """Given epoch, step, and parameters, compute metric using the required params (e.g. using `params_dict`).

        Required params are set up in `add_params` from the METRIC_PARAMS key from the configuration.

        Args:
            epoch (int): The epoch to save this update under
            step (int): The step to save this update under
            params (Dict[str, Any]): The parameters provided by the metric calling object

        Returns:
            bool: Update success
        """
        required_params = self._get_required_params(params=params)
        response = self._compute_metric(epoch, step, **required_params)
        success = self._add_value(epoch, step, response)
        return success

    @abstractmethod
    def _get_required_params(self, params: Dict[str, Any]) -> dict[str, Any]:
        """Obtain the subset of required parameters from the `params` dictionary provided by the metrics caller.

        Required parameters are provided in the METRIC_PARAMS key in the metric's configuration.

        Args:
            params (Dict[str, Any]): The parameters provided by the metric calling object

        Returns:
            Dict[str, Any]: The subset of required parameters, mapped to the parameter names in `self.params_dict`
        """
        return {key:params[self.params_dict[key]] for key in self.params_dict}

    @abstractmethod
    def _compute_metric(self, epoch: int, step: int, **kwargs) -> float:
        """Compute the actual metric value for the provided epoch, step, and kwargs (i.e. params). kwargs can be overridden with desired parameters.

        For example, Accuracy metric's desired kwargs are `preds` and `targets`.

        Args:
            epoch (int): The current epoch when this metric's compute is called
            step (int): The current step when this metric's compute is called

        Raises:
            NotImplementedError: _description_

        Returns:
            float: The compute metric value.
        """
        raise NotImplementedError()

    @abstractmethod
    def _add_value(self, epoch: int, step: int, metric_value: float) -> bool:
        """Add the provided metric value at the provided epoch-step pair into memory. 

        Later, we can batch save a group of computed metrics into backend storage or some metrics server.

        Args:
            epoch (int): Epoch in this save triplet.
            step (int): Step in this save triplet
            metric_value (float): The computed metric value in this save triplet

        Returns:
            bool: Success in adding the value. 
        """
        self.memory.append((epoch, step, metric_value))
        return True

    def clear(self, **kwargs):
        """Clear/reset object computation memory as well as state."""
        self._clear_state()
        self._clear_memory()
    
    @abstractmethod
    def _clear_state(self, **kwargs):
        self.state = {}

    @abstractmethod
    def _clear_memory(self, **kwargs):
        self.memory = []

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
    #   We can create another top-level framework inherited from BaseTrainer for ease of use, but basically, it's all in the metrics itself
    # each metric is tied to an individual storage, to collect different types of metrics. THis is managed in MetricsManager section of config
    #   This is fine-grained and coarse. We can individually determine each metric's storage, and give categories, likle logmetrics, their own storage for every logmetrics...
    #   Haven't quiet decided on how this would be implemented.

    def _serialize(self, **kwargs) -> Tuple[bool, List[Tuple[str, str, str, int, int, float]]]:  # [(metric_name, metric_type, metric_class, epoch, step, metric_value)]
        """Returns a list of tuples containing metrics saved in memory so far, converted into a string

        Raises:
            NotImplementedError: _description_

        Returns:
            List[Tuple[str, str, str, int, int, float]]: [(metric_name, metric_type, metric_class, epoch, step, metric_value)]. This is combined with ers-key in the storage.
        """
        raise NotImplementedError()

    def _serialize_to_string(self, delimiter = ",", **kwargs)-> Tuple[bool, Any]:
        """Writes metrics saved so far in memory to a string, one entry per line, and yields the string

        Args:
            delimiter (str, optional): _description_. Defaults to ",".

        Returns:
            bool: _description_
        """
        raise NotImplementedError()

    def _noop_serialize(self, **kwargs)-> Tuple[bool, Any]:
        """A dummy serializer to do nothing.

        Returns:
            _type_: _description_
        """
        return (True,None)


    def _save(self, **kwargs):
        """Metrics own saving method. Must be implemented is `will_save_itself` is True

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def _storage_save(self, **kwargs):
        """Metric's method to save to storage. Must be implemented if `metric_storage` is not None

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def _metric_save(self, **kwargs):
        """The standard save. Metric will serialize to stream or file for MetricsManager.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

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