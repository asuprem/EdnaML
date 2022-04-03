from ednaml.datareaders import DataReader
from ednaml.crawlers import CoLabelCompCarsCrawler
from ednaml.generators.ClassificationGenerator import (
    ClassificationDataset,
    ClassificationGenerator,
)


class CompCars(DataReader):
    name: str = "CompCars"
    CRAWLER = CoLabelCompCarsCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__()
