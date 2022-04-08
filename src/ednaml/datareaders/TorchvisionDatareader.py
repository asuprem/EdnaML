



from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader


class TorchvisionDatareader(DataReader):
    name: str = "TorchvisionDatareader"
    CRAWLER = Crawler
    DATASET = None
    GENERATOR = None