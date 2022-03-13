
from . import DataReader
from crawlers import VehicleIDDataCrawler
from generators.ClassificationGenerator import ClassificationDataset, ClassificationGenerator

class VehicleID(DataReader):
    CRAWLER = VehicleIDDataCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__() 