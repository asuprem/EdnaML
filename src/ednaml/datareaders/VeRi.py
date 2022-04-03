from ednaml.datareaders import DataReader
from crawlers import VeRiDataCrawler
from generators.ClassificationGenerator import (
    ClassificationDataset,
    ClassificationGenerator,
)


class VeRi(DataReader):
    name: str = "VeRi"
    CRAWLER = VeRiDataCrawler
    DATASET = ClassificationDataset
    GENERATOR = ClassificationGenerator

    def __init__(self):
        super().__init__()
