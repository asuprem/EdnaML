
from . import DataReader
from crawlers import VeRiDataCrawler
from generators.ClassificationGenerator import ClassificationDataset, ClassificationGenerator

class VeRi(DataReader):
    CRAWLER = VeRiDataCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__() 