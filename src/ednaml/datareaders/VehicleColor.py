from ednaml.datareaders import DataReader
from ednaml.crawlers import VehicleColorCrawler
from ednaml.generators.ClassificationGenerator import (
    ClassificationDataset,
    ClassificationGenerator,
)


class VehicleColor(DataReader):
    name: str = "VehicleColors"
    CRAWLER = VehicleColorCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__()
