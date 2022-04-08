



from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators.TorchvisionGeneratorWrapper import TorchvisionGeneratorWrapper


class TorchvisionDatareader(DataReader):
    name: str = "TorchvisionDatareader"
    GENERATOR = TorchvisionGeneratorWrapper