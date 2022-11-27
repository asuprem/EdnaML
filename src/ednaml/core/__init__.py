from typing import Any, Dict, Type

from ednaml.crawlers import Crawler
from ednaml.generators import Generator
import logging
import ednaml.core.decorators

class EdnaMLBase:
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
        ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS = {}

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

from ednaml.core.EdnaML import EdnaML
from ednaml.core.EdnaDeploy import EdnaDeploy
