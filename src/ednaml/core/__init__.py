import logging
from types import MethodType
from typing import Any, Dict, List, Type, Union
import torch
from torchinfo import ModelStatistics
import ednaml.core.decorators as edna
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.core import EdnaMLContextInformation
from ednaml.crawlers import Crawler
from ednaml.generators import Generator
from ednaml.logging import LogManager
from ednaml.loss.builders import LossBuilder
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.plugins.ModelPlugin import ModelPlugin
from ednaml.storage import BaseStorage, StorageManager
from ednaml.trainer.BaseTrainer import BaseTrainer
from ednaml.utils import ExperimentKey
from ednaml.utils.LabelMetadata import LabelMetadata


class EdnaMLBase:
    labelMetadata: LabelMetadata
    modelStatistics: ModelStatistics
    model: ModelAbstract
    loss_function_array: List[LossBuilder]
    loss_optimizer_array: List[torch.optim.Optimizer]
    optimizer: List[torch.optim.Optimizer]
    scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    trainer: BaseTrainer
    crawler: Crawler
    train_generator: Generator
    test_generator: Generator
    cfg: EdnaMLConfig
    config: Union[str,List[str]]    # list of paths to configuration files
    decorator_reference: Dict[str,Type[MethodType]]
    plugins: Dict[str, ModelPlugin]
    storage: Dict[str, BaseStorage]
    storage_classes: Dict[str, Type[BaseStorage]]
    storageManager: StorageManager
    logManager: LogManager
    logger: logging.Logger
    experiment_key: ExperimentKey

    context_information: EdnaMLContextInformation
    logLevels = {
        0: logging.NOTSET,
        1: logging.ERROR,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    logger: logging.Logger
    _crawlerClassQueue: Type[Crawler]
    _crawlerArgsQueue: Dict[str, Any]
    _crawlerClassQueueFlag: bool
    _crawlerInstanceQueue: Crawler
    _crawlerInstanceQueueFlag: bool
    _generatorClassQueue = Type[Generator]
    _generatorArgsQueue = Dict[str, Any]
    _generatorClassQueueFlag = bool

    @staticmethod
    def clear_registrations():
        edna.REGISTERED_EDNA_COMPONENTS = {}

    def log(self, msg):
        self.logger.info("[EdnaML]" + msg)
    def debug(self, msg):
        self.logger.debug("[EdnaML]" + msg)

    def __init__(self):
        self.logger = None
        self.model = None
        self.plugins = {}
        self.storage = {}
        self.storage_classes = {}
        self.storageManager = None
        self.logManager = None
        self.pretrained_weights = None
class EdnaMLContextInformation:
    MODEL_HAS_LOADED_WEIGHTS: bool = False
    LOADED_EPOCH: int = -1
    LOADED_STEP: int = -1

from ednaml.core.EdnaDeploy import EdnaDeploy
from ednaml.core.EdnaML import EdnaML
