from ednaml.datareaders import DataReader
from ednaml.generators.AlbertGenerator import AlbertGenerator, AlbertDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AlbertReader(DataReader):
    name: str = "AlbertReader"
    DATASET: Dataset = AlbertDataset
    GENERATOR = AlbertGenerator
