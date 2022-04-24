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

from typing import Any, Dict, List, Tuple, Union
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

import ednaml.utils
from ednaml.utils.LabelMetadata import LabelMetadata

class Generator:
    num_entities: LabelMetadata

    def __init__(
        self,
        gpus: int,
        transforms: Dict[str,Any],
        **kwargs
    ):
        """Initialize the generator
        """
        raise NotImplementedError()

    # NOTE removed instance parameter from here.,, is it needed???
    def build(self, datacrawler, mode, batch_size, workers, **kwargs):
        """This should generate a TorchDataset and associated DataLoader to yield batches.

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
        self, datacrawler, mode: str, transform: List[object], **kwargs) -> TorchDataset:
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


from ednaml.generators.ImageGenerator import ImageGenerator
from ednaml.generators.TextGenerator import TextGenerator

from ednaml.generators.ClassificationGenerator import ClassificationGenerator
from ednaml.generators.MultiClassificationGenerator import MultiClassificationGenerator
from ednaml.generators.TorchvisionGeneratorWrapper import TorchvisionGeneratorWrapper
