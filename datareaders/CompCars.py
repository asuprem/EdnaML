
from . import DataReader
from crawlers import CoLabelCompCarsCrawler
from generators.ClassificationGenerator import ClassificationDataset, ClassificationGenerator

class CompCars(DataReader):
    CRAWLER = CoLabelCompCarsCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__() 