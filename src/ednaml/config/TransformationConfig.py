from typing import List
from ednaml.config import BaseConfig

from ednaml.config.ConfigDefaults import ConfigDefaults


class TransformationConfig(BaseConfig):
    SHAPE: List[int]
    NORMALIZATION_MEAN: List[int]
    NORMALIZATION_STD: List[int]
    NORMALIZATION_SCALE: int
    H_FLIP: float
    T_CROP: bool
    RANDOM_ERASE: bool
    RANDOM_ERASE_VALUE: float
    CHANNELS: int
    BATCH_SIZE: int
    WORKERS: int

    def __init__(self, transformation_dict, defaults: ConfigDefaults):
        self.SHAPE = transformation_dict.get("SHAPE", defaults.SHAPE)
        self.NORMALIZATION_MEAN = transformation_dict.get(
            "NORMALIZATION_MEAN", defaults.NORMALIZATION_MEAN
        )
        self.NORMALIZATION_STD = transformation_dict.get(
            "NORMALIZATION_STD", defaults.NORMALIZATION_STD
        )
        self.NORMALIZATION_SCALE = transformation_dict.get(
            "NORMALIZATION_SCALE", defaults.NORMALIZATION_SCALE
        )
        self.H_FLIP = transformation_dict.get("H_FLIP", defaults.H_FLIP)
        self.T_CROP = transformation_dict.get("T_CROP", defaults.T_CROP)
        self.RANDOM_ERASE = transformation_dict.get(
            "RANDOM_ERASE", defaults.RANDOM_ERASE
        )
        self.RANDOM_ERASE_VALUE = transformation_dict.get(
            "RANDOM_ERASE_VALUE", defaults.RANDOM_ERASE_VALUE
        )
        self.CHANNELS = transformation_dict.get("CHANNELS", defaults.CHANNELS)
        self.BATCH_SIZE = transformation_dict.get("BATCH_SIZE", defaults.BATCH_SIZE)
        self.WORKERS = transformation_dict.get("WORKERS", defaults.WORKERS)
