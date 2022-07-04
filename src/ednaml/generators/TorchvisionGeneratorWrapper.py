from typing import List, Tuple, Union
from ednaml.generators import ImageGenerator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from ednaml.utils import locate_class
from torchvision.datasets import VisionDataset


class TorchvisionGeneratorWrapper(ImageGenerator):
    """Generator for a TorchVision dataset.

    Args:
        ImageGenerator (_type_): _description_
    """

    def __init__(
        self,
        logger, 
        gpus: int,
        i_shape: Union[List[int], Tuple[int, int]],
        channels: int,
        normalization_mean: float,
        normalization_std: float,
        normalization_scale: float,
        **kwargs
    ):
        super().__init__(logger,
            gpus,
            i_shape,
            channels,
            normalization_mean,
            normalization_std,
            normalization_scale,
            **kwargs
        )

        """
        So we will have arguments inside the kwargs that set up the dataloader object
        """
        self.gpus = gpus
        self.torchvision_dataset_class = kwargs.get("tv_dataset")
        self.torchvision_dataset_args = kwargs.get("tv_args")

    def build(self, datacrawler, mode, batch_size, workers, **kwargs):
        # We don't need the data crawler...

        dataset_class: VisionDataset = locate_class(
            package="torchvision",
            subpackage="datasets",
            classpackage=self.torchvision_dataset_class,
        )
        # We probably need something here to adjust the arguments for each dataset class, because of differences in arguments...

        # dataset API is not unified, see discussions below
        # https://github.com/pytorch/vision/issues/1067
        # https://github.com/pytorch/vision/issues/1080
        if self.torchvision_dataset_class in ["CIFAR10", "MNIST", "USPS"]:
            # These have a train argument...
            self.dataset = dataset_class(
                root=self.torchvision_dataset_args.get("root"),
                train=(mode == "train"),
                transform=self.transformer,
                target_transform=None,
                **self.torchvision_dataset_args
            )
        elif self.torchvision_dataset_class in [
            "CelebA",
            "SVHN",
            "STL10",
            "ImageNet",
            "Cityscapes",
        ]:
            if self.torchvision_dataset_class == "ImageNet":
                mode = mode if mode == "train" else "val"
            self.dataset = dataset_class(
                root=self.torchvision_dataset_args.get("root"),
                split=mode,
                transform=self.transformer,
                target_transform=None,
                **self.torchvision_dataset_args
            )
        else:
            raise NotImplementedError()

        self.dataloader = self.buildDataLoader(
            self.dataset, mode, batch_size=batch_size, **kwargs
        )

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=kwargs.get("shuffle", True),
            num_workers=self.workers,
        )
