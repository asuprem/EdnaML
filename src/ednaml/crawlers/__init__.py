import os
from typing import Dict


class Crawler:
    """The base crawler class. TODO"""

    classes: Dict[str, int]

    def __init__(self):
        self.classes = {}

    def __verify(self, folder):
        if not os.path.exists(folder):
            raise IOError(
                "Folder {data_folder} does not exist".format(data_folder=folder)
            )
        else:
            self.logger.info("Found {data_folder}".format(data_folder=folder))


# Vehicle Re-ID Crawlers
from ednaml.crawlers.VeRiDataCrawler import VeRiDataCrawler
from ednaml.crawlers.VRICDataCrawler import VRICDataCrawler
from ednaml.crawlers.VehicleIDDataCrawler import VehicleIDDataCrawler

# Carzam crawlers
from ednaml.crawlers.Cars196DataCrawler import Cars196DataCrawler

# CoLabelCrawlers
from ednaml.crawlers.VehicleColorCrawler import VehicleColorCrawler
from ednaml.crawlers.CoLabelIntegratedDatasetCrawler import (
    CoLabelIntegratedDatasetCrawler,
)

KnowledgeIntegratedDatasetCrawler = CoLabelIntegratedDatasetCrawler

from ednaml.crawlers.CoLabelCompCarsCrawler import CoLabelCompCarsCrawler
