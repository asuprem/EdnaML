from ednaml.datareaders import DataReader
from crawlers import VehicleIDDataCrawler
from generators.ClassificationGenerator import (
    ClassificationDataset,
    ClassificationGenerator,
)


class VehicleID(DataReader):
    name: str = "VehicleID"
    CRAWLER = VehicleIDDataCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__()
