from typing import List, Tuple, Union
from ednaml.generators import ImageGenerator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from ednaml.utils import locate_class
from torchvision.datasets import VisionDataset
import torch
from ednaml.utils.LabelMetadata import LabelMetadata


class TorchvisionGeneratorWrapper(ImageGenerator):
    """Generator for a TorchVision dataset.

    Args:
        ImageGenerator (_type_): _description_
    """

    def __init__(
        self, logger=None, gpus: int = 1, transforms={}, mode: str = "train", **kwargs
    ):
        super().__init__(logger, gpus=gpus, transforms=transforms, mode=mode, **kwargs)

        

    def buildGeneratorAttributes(self, **kwargs):
        self.torchvision_dataset_class = kwargs.get("tv_dataset")
        self.torchvision_dataset_args = kwargs.get("tv_args")

    def buildDataset(self, datacrawler, mode: str, transform: List[object], **kwargs) -> TorchDataset:

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
            return dataset_class(
                root=self.torchvision_dataset_args.get("root"),
                train=(mode == "train"),
                transform=self.transformer,
                target_transform=None,
                **self.torchvision_dataset_args.get("args", {})
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
            return dataset_class(
                root=self.torchvision_dataset_args.get("root"),
                split=mode,
                transform=self.transformer,
                target_transform=None,
                **self.torchvision_dataset_args
            )
        else:
            raise NotImplementedError()


    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=kwargs.get("shuffle", True),
            num_workers=self.workers,
        )

    def getNumEntities(self, datacrawler, mode, **kwargs) -> LabelMetadata:
        label_name = kwargs.get("label_name", "label")
        num_classes = kwargs.get("num_classes", torch.unique(self.dataset.targets).shape[0])
        return LabelMetadata({label_name: {"classes": num_classes}})
