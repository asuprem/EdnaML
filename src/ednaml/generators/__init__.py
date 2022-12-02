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

from logging import Logger
from typing import Any, Dict, List
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from ednaml.utils.LabelMetadata import LabelMetadata


class Generator:
    num_entities: LabelMetadata
    dataloader: TorchDataLoader
    dataset: TorchDataset
    gpus: int
    mode: str
    training_mode: bool

    def __init__(
        self,
        logger: Logger,
        gpus: int = 1,
        transforms: Dict[str, Any] = {},
        mode: str = "train",
        **kwargs
    ):
        """Initialize the generator"""
        self.logger = logger
        self.gpus = max(1, gpus)
        self.dataloader = None
        self.transforms = transforms
        self.mode = mode
        self.training_mode = self.isTrainingMode()
        self.transformer = self.build_transforms(self.transforms, self.mode, **kwargs)
        self.buildGeneratorAttributes(**kwargs)

    def buildGeneratorAttributes(self, **kwargs):
        pass

    def build_transforms(self, transforms: Dict[str, Any], mode, **kwargs):
        return None

    def isTrainingMode(self):
        return self.mode == "train"

    def build(self, datacrawler, batch_size, workers, **kwargs):
        """This should generate a TorchDataset and associated DataLoader to yield batches.

        Raises:
            NotImplementedError: _description_
        """

        self.workers = workers * self.gpus

        self.dataset = self.buildDataset(
            datacrawler, self.mode, self.transformer, **kwargs
        )
        self.dataloader = self.buildDataLoader(
            self.dataset, self.mode, batch_size=batch_size, **kwargs
        )
        self.num_entities = self.getNumEntities(datacrawler, self.mode, **kwargs)

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
        return None

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        """Given a Torch Dataset, build a TorchDataloader to return a tensor, possibly with a collate function

        Args:
            dataset (_type_): _description_
            batch_size (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """

        return None

    def getNumEntities(self, datacrawler, mode, **kwargs) -> LabelMetadata:
        """Return the number of classes in the overall label scheme
        Raises:
            NotImplementedError: _description_
        """

        return None


from ednaml.generators.ImageGenerator import ImageGenerator
from ednaml.generators.TextGenerator import TextGenerator

from ednaml.generators.ClassificationGenerator import ClassificationGenerator
from ednaml.generators.MultiClassificationGenerator import (
    MultiClassificationGenerator,
)
from ednaml.generators.TorchvisionGeneratorWrapper import (
    TorchvisionGeneratorWrapper,
)
from ednaml.generators.HFGenerator import HFGenerator
