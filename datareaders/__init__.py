
from crawlers import Crawler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from generators import ImageGenerator
class DataReader:
    CRAWLER: Crawler = None
    DATASET: Dataset = None
    DATALOADER: DataLoader = None
    GENERATOR: ImageGenerator = None

    def __init__(self):
        pass

from .VehicleColor import VehicleColor
from .CompCars import CompCars