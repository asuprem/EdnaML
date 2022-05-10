from sched import scheduler
from typing import Dict
from ednaml.config import BaseConfig

from ednaml.config.ConfigDefaults import ConfigDefaults


class SchedulerConfig(BaseConfig):
    SCHEDULER_NAME: str
    LR_SCHEDULER: str
    LR_KWARGS: Dict[str, str]

    def __init__(self, scheduler_dict, defaults: ConfigDefaults):
        self.SCHEDULER_NAME = scheduler_dict.get(
            "SCHEDULER_NAME", defaults.SCHEDULER_NAME
        )
        self.LR_SCHEDULER = scheduler_dict.get(
            "LR_SCHEDULER", defaults.LR_SCHEDULER
        )
        self.LR_KWARGS = scheduler_dict.get("LR_KWARGS", defaults.LR_KWARGS)
