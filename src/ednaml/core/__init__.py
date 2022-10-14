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


from ednaml.core.EdnaML import EdnaML
from ednaml.core.EdnaDeploy import EdnaDeploy
