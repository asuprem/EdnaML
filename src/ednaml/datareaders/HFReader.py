from ednaml.datareaders import DataReader
from ednaml.generators.HFGenerator import HFGenerator, HFDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class HFReader(DataReader):
    name: str = "HFReader"
    DATASET: Dataset = HFDataset
    GENERATOR = HFGenerator
