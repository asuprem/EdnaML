from typing import Any, Dict, Type

from ednaml.crawlers import Crawler


class EdnaMLBase:
    _crawlerClassQueue: Type[Crawler]
    _crawlerArgsQueue: Dict[str,Any]
    _crawlerClassQueueFlag: bool
    _crawlerInstanceQueue: Crawler
    _crawlerInstanceQueueFlag: bool


from ednaml.core.EdnaML import EdnaML


