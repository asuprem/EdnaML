# Vehicle Re-ID Crawlers
from .VeRiDataCrawler import VeRiDataCrawler
from .VRICDataCrawler import VRICDataCrawler

# Carzam crawlers
from .Cars196DataCrawler import Cars196DataCrawler

# Other Crawlers
from .ClassedCrawler import ClassedCrawler


# CoLabelCrawlers
from .CoLabelVehicleColorCrawler import CoLabelVehicleColorCrawler
from .CoLabelIntegratedDatasetCrawler import CoLabelIntegratedDatasetCrawler
KnowledgeIntegratedDatasetCrawler = CoLabelIntegratedDatasetCrawler

import os
class Crawler:
    def __init__():
        raise NotImplementedError()

    def __verify(self,folder):
        if not os.path.exists(folder):
            raise IOError("Folder {data_folder} does not exist".format(data_folder=folder))
        else:
            self.logger.info("Found {data_folder}".format(data_folder = folder))
