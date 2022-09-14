import tqdm, json
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
import ednaml.loss.builders
from typing import List
from ednaml.crawlers import Crawler
from ednaml.trainer import BaseTrainer
from ednaml.utils.LabelMetadata import LabelMetadata



class ClassificationTrainer(BaseTrainer):

    def step(self, batch) -> torch.Tensor:
        return super().step(batch)
    
    def evaluate_impl(self):
        return super().evaluate_impl()

    