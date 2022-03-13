
from . import DataReader
from crawlers import CoLabelVehicleColorCrawler
from generators.ClassificationGenerator import ClassificationDataset, ClassificationGenerator

class VehicleColor(DataReader):
    CRAWLER = CoLabelVehicleColorCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__() 