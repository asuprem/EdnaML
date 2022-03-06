
from . import DataReader
from ..crawlers import CoLabelVehicleColorCrawler
from ..generators.CoLabelGenerator import CoLabelDataset, CoLabelGenerator

class VehicleColor(DataReader):
    CRAWLER = CoLabelVehicleColorCrawler
    DATASET = CoLabelDataset
    GENERATOR = CoLabelGenerator

    def __init__(self):
        super().__init__() 