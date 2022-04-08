from ednaml.datareaders import DataReader
from ednaml.crawlers import VeRiDataCrawler
from ednaml.generators.ClassificationGenerator import (
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
