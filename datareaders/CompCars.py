
from . import DataReader
from crawlers import CoLabelCompCarsCrawler
from generators.CoLabelGenerator import CoLabelDataset, CoLabelGenerator

class CompCars(DataReader):
    CRAWLER = CoLabelCompCarsCrawler
    DATASET = CoLabelDataset
    GENERATOR = CoLabelGenerator

    def __init__(self):
        super().__init__() 