from ednaml.generators import Generator
from ednaml.utils.LabelMetadata import LabelMetadata
from typing import List, Tuple, Union
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
import ednaml.utils


class ImageGenerator(Generator):
    """Base class for image dataset generators"""

    def __init__(
        self, logger=None, gpus: int = 1, transforms={}, mode: str = "train", **kwargs
    ):
        """Initializes the Generator and builds the data transformer

        Args:
            gpus (_type_): _description_
            i_shape (_type_): _description_
            normalization_mean (_type_): _description_
            normalization_std (_type_): _description_
            normalization_scale (_type_): _description_
        """
        super().__init__(logger, gpus=gpus, transforms=transforms, mode=mode, **kwargs)

    def build_transforms(self, transforms, mode, **kwargs):
        return T.Compose(self._build_transforms(**transforms.ARGS))

    def _build_transforms(
        self,
        i_shape: Union[List[int], Tuple[int, int]],
        channels: int,
        normalization_mean: float,
        normalization_std: float,
        normalization_scale: float,
        **kwargs
    ) -> List[object]:
        """Builds the transforms for the images in dataset. This can be replaced for custom set of transforms

        Args:
            i_shape (Union[List[int,int],Tuple[int,int]]): _description_
            normalization_mean (float): _description_
            normalization_std (float): _description_
            normalization_scale (float): _description_

        Returns:
            _type_: _description_
        """
        if (
            i_shape == []
            and channels == 0
            and normalization_mean == 0
            and normalization_std == 0
            and normalization_scale == 0
        ):
            return []
        if type(normalization_mean) is not list:
            (normalization_mean, normalization_std,) = ednaml.utils.extend_mean_arguments(
                [normalization_mean, normalization_std], channels
            )
        if len(normalization_mean) != channels:
            (normalization_mean, normalization_std,) = ednaml.utils.extend_mean_arguments(
                [normalization_mean[0], normalization_std[0]], channels
            )
        transformer_primitive = []
        transformer_primitive.append(T.Resize(size=i_shape))
        if kwargs.get("h_flip", 0) > 0:
            transformer_primitive.append(T.RandomHorizontalFlip(p=kwargs.get("h_flip")))
        if kwargs.get("t_crop", False):
            transformer_primitive.append(T.RandomCrop(size=i_shape))
        transformer_primitive.append(T.ToTensor())
        transformer_primitive.append(
            T.Normalize(mean=normalization_mean, std=normalization_std)
        )
        if kwargs.get("rea", False):
            transformer_primitive.append(
                T.RandomErasing(
                    p=0.5, scale=(0.02, 0.4), value=kwargs.get("rea_value", 0)
                )
            )
        return transformer_primitive

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

    def getNumEntities(self, datacrawler, mode, **kwargs) -> LabelMetadata:
        """Return the number of classes in the overall label scheme
        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError()
