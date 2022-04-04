#!/usr/bin/env python3
"""Generators yields batches to models during training and testing. 
Requirs a crawler to provide a list of paths with labels.
Generator then loads a random subset, performs transformations, 
and yields these to model when requested.

List of supported generators:

    - SequencedGenerator: Yields a triplet batch for re-id
    - TripletGenerator: Yields a triplet batch for re-id
    - ClassedGenerator:
    - Cars196Generator:
    - ClassificationGenerator:
    - CoLabelIntegratedDatasetGenerator: 
    - CoLabelDeployGenerator:
    - KnowledgeIntegratedGenerator: 
"""

from typing import Any, List, Tuple, Union
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

import ednaml.utils
from ednaml.utils.LabelMetadata import LabelMetadata


class ImageGenerator:
    """Base class for image dataset generators
    """

    num_entities: LabelMetadata

    def __init__(
        self,
        gpus: int,
        i_shape: Union[List[int], Tuple[int, int]],
        channels: int,
        normalization_mean: float,
        normalization_std: float,
        normalization_scale: float,
        **kwargs
    ):
        """Initializes the Generator and builds the data transformer

        Args:
            gpus (_type_): _description_
            i_shape (_type_): _description_
            normalization_mean (_type_): _description_
            normalization_std (_type_): _description_
            normalization_scale (_type_): _description_
        """
        self.gpus = gpus
        self.transformer = T.Compose(
            self.build_transforms(
                i_shape,
                channels,
                normalization_mean,
                normalization_std,
                normalization_scale,
                **kwargs
            )
        )

    def build_transforms(
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
        normalization_mean, normalization_std = ednaml.utils.extend_mean_arguments(
            [normalization_mean, normalization_std], channels
        )
        transformer_primitive = []
        transformer_primitive.append(T.Resize(size=i_shape))
        if kwargs.get("h_flip") > 0:
            transformer_primitive.append(T.RandomHorizontalFlip(p=kwargs.get("h_flip")))
        if kwargs.get("t_crop"):
            transformer_primitive.append(T.RandomCrop(size=i_shape))
        transformer_primitive.append(T.ToTensor())
        transformer_primitive.append(
            T.Normalize(mean=normalization_mean, std=normalization_std)
        )
        if kwargs.get("rea"):
            transformer_primitive.append(
                T.RandomErasing(
                    p=0.5, scale=(0.02, 0.4), value=kwargs.get("rea_value", 0)
                )
            )
        return transformer_primitive

    # NOTE removed instance parameter from here.,, is it needed???
    def setup(self, datacrawler, mode, batch_size, workers, **kwargs):
        """This should generate a TorchDataset and associated DataLoader to yield batches.
        The actual steps are as follows:

        Raises:
            NotImplementedError: _description_
        """

        self.workers = workers * self.gpus

        self.dataset = self.buildDataset(datacrawler, mode, self.transformer, **kwargs)
        self.dataloader = self.buildDataLoader(
            self.dataset, mode, batch_size=batch_size, **kwargs
        )
        self.num_entities = self.getNumEntities(datacrawler, mode, **kwargs)

    def buildDataset(
        self, datacrawler, mode: str, transform: List[object], **kwargs
    ) -> TorchDataset:
        """Given the daatacrawler with all the data, and the mode (could be 
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


from ednaml.generators.ClassificationGenerator import ClassificationGenerator
from ednaml.generators.CoLabelIntegratedDatasetGenerator import CoLabelIntegratedDatasetGenerator
from ednaml.generators.CoLabelDeployGenerator import CoLabelDeployGenerator
from ednaml.generators.MultiClassificationGenerator import MultiClassificationGenerator

KnowledgeIntegratedGenerator = CoLabelIntegratedDatasetGenerator
