
from . import DataReader
from crawlers import VeRiDataCrawler
from generators.CoLabelGenerator import CoLabelDataset, CoLabelGenerator

class VeRi(DataReader):
    CRAWLER = VeRiDataCrawler
    DATASET = CoLabelDataset
    GENERATOR = CoLabelGenerator

    def __init__(self):
        super().__init__() 