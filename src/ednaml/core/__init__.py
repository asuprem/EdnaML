from typing import Any, Dict, Type

from ednaml.crawlers import Crawler
from ednaml.generators import Generator


class EdnaMLBase:
    _crawlerClassQueue: Type[Crawler]
    _crawlerArgsQueue: Dict[str,Any]
    _crawlerClassQueueFlag: bool
    _crawlerInstanceQueue: Crawler
    _crawlerInstanceQueueFlag: bool
    _generatorClassQueue = Type[Generator]
    _generatorArgsQueue = Dict[str,Any]
    _generatorClassQueueFlag = bool


from ednaml.core.EdnaML import EdnaML


