from ednaml.crawlers import Crawler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ednaml.generators import ImageGenerator


class DataReader:
    name: str = "DataReader"
    CRAWLER: Crawler = Crawler
    DATASET: Dataset = Dataset
    DATALOADER: DataLoader = DataLoader
    GENERATOR: ImageGenerator = ImageGenerator

    def __init__(self):
        pass


"""
from .VehicleColor import VehicleColor
from .CompCars import CompCars
from .VehicleID import VehicleID
from .VeRi import VeRi
"""
