



from typing import List, Tuple, Union
from ednaml.generators import ImageGenerator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from ednaml.utils import locate_class

class TorchvisionGeneratorWrapper(ImageGenerator):
    """Generator for a TorchVision dataset.

    Args:
        ImageGenerator (_type_): _description_
    """


    def __init__(self, gpus: int, i_shape: Union[List[int], Tuple[int, int]], channels: int, normalization_mean: float, normalization_std: float, normalization_scale: float, **kwargs):
        super().__init__(gpus, i_shape, channels, normalization_mean, normalization_std, normalization_scale, **kwargs)

        """
        So we will have arguments inside the kwargs that set up the dataloader object
        """
        self.gpus = gpus
        self.torchvision_dataset_class = kwargs.get("tv_dataset")
        self.torchvision_dataset_args = kwargs.get("tv_args")

    def setup(self, datacrawler, mode, batch_size, workers, **kwargs):
        # We don't need the data crawler...

        dataset_class = locate_class(package="torchvision", subpackage="datasets", classpackage=self.torchvision_dataset_class)
        # We probably need something here to adjust the arguments for each dataset class, because of differences in arguments... 
