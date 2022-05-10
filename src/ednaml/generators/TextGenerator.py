from typing import Any, List, Tuple, Union
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
from ednaml.generators import Generator

import ednaml.utils
from ednaml.utils.LabelMetadata import LabelMetadata


class TextGenerator(Generator):
    """Base class for text generators"""

    def __init__(
        self, gpus: int = 1, transforms={}, mode: str = "train", **kwargs
    ):
        """Initializes the Generator and builds the data transformer

        Args:
            gpus (_type_): _description_
            i_shape (_type_): _description_
            normalization_mean (_type_): _description_
            normalization_std (_type_): _description_
            normalization_scale (_type_): _description_
        """
        super().__init__(gpus=gpus, transforms=transforms, mode=mode, **kwargs)

    def buildDataset(
        self, datacrawler, mode: str, transform: List[object], **kwargs
    ) -> TorchDataset:
        """Given the datacrawler with all the data, and the mode (could be
        any user-defined mode such as 'train', 'test', 'zsl', 'gzsl', etc), as
        as well the transform, return a TorchDataset

        Args:
            datacrawler (_type_): _description_
            mode (_type_): _description_
            transform (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        """Given a Torch Dataset, build a TorchDataloader to return a tensor, possibly with a collate function

        Args:
            dataset (_type_): _description_
            batch_size (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError()
