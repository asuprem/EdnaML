from typing import List
from ednaml.config import BaseConfig

from ednaml.config.ConfigDefaults import ConfigDefaults


class TransformationConfig(BaseConfig):
    #batch size, workers and args should be used only!!!
    #SHAPE: List[int]
    #NORMALIZATION_MEAN: List[int]
    #NORMALIZATION_STD: List[int]
    #NORMALIZATION_SCALE: int
    #H_FLIP: float
    #T_CROP: bool
    #RANDOM_ERASE: bool
    #RANDOM_ERASE_VALUE: float
    #CHANNELS: int
    BATCH_SIZE: int
    WORKERS: int

    def __init__(self, transformation_dict, defaults: ConfigDefaults): #all of this will go, as shape mean etc won't matter
        #only batch size, and workers are required
        #defaults empty ()
        #
        print("transformation_dict ::: ",transformation_dict)
        '''self.shape = transformation_dict.get("shape", defaults.SHAPE)
        self.normalization_mean = transformation_dict.get(
            "normalization_mean", defaults.NORMALIZATION_MEAN
        )
        self.normalization_std = transformation_dict.get(
            "normalization_std", defaults.NORMALIZATION_STD
        )
        self.normalization_scale = transformation_dict.get(
            "normalization_scale", defaults.NORMALIZATION_SCALE
        )
        self.h_flip = transformation_dict.get("h_flip", defaults.H_FLIP)
        self.t_crop = transformation_dict.get("t_crop", defaults.T_CROP)
        self.RANDOM_ERASE = transformation_dict.get(
            "RANDOM_ERASE", defaults.RANDOM_ERASE
        )
        self.RANDOM_ERASE_VALUE = transformation_dict.get(
            "RANDOM_ERASE_VALUE", defaults.RANDOM_ERASE_VALUE
        )
        self.CHANNELS = transformation_dict.get("CHANNELS", defaults.CHANNELS)'''
        self.BATCH_SIZE = transformation_dict.get(
            "BATCH_SIZE", defaults.BATCH_SIZE
        )
        self.WORKERS = transformation_dict.get("WORKERS", defaults.WORKERS)
        self.ARGS = transformation_dict.get("ARGS", {})
        
