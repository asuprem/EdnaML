
from . import DataReader
from crawlers import VehicleIDDataCrawler
from generators.CoLabelGenerator import CoLabelDataset, CoLabelGenerator

class VehicleID(DataReader):
    CRAWLER = VehicleIDDataCrawler
    DATASET = CoLabelDataset
    GENERATOR = CoLabelGenerator

    def __init__(self):
        super().__init__() 