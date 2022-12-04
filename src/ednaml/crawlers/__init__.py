import os
from typing import Dict


class Crawler:
    """The base crawler class. TODO"""

    classes: Dict[str, int]

    def __init__(self, logger, **kwargs):
        self.classes = {}

    def __verify__(self, folder):
        if not os.path.exists(folder):
            raise IOError(
                "Folder {data_folder} does not exist".format(data_folder=folder)
            )
        else:
            self.logger.info("Found {data_folder}".format(data_folder=folder))
