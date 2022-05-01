from typing import Dict, List

from ednaml.config import BaseConfig


class ConfigDefaults(BaseConfig):
    OPTIMIZER_BUILDER: str
    MODEL_SERVING: bool
    EPOCHS: int
    SKIPEVAL: bool
    TEST_FREQUENCY: int
    TRAINER: str
    MODEL_VERSION: int
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    DRIVE_BACKUP: bool
    SAVE_FREQUENCY: int
    CHECKPOINT_DIRECTORY: str

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

    BUILDER: str
    MODEL_ARCH: str
    MODEL_BASE: str
    MODEL_NORMALIZATION: str
    MODEL_KWARGS: Dict[str, str]
    PARAMETER_GROUPS: List[str]

    LOSSES: List[str]
    LOSS_KWARGS: List[Dict[str, str]]
    LAMBDAS: List[int]
    LOSS_LABEL: str
    LOSS_NAME: str

    OPTIMIZER_NAME: str
    OPTIMIZER_KWARGS: Dict[str, str]
    BASE_LR: float
    LR_BIAS_FACTOR: float
    WEIGHT_DECAY: float
    WEIGHT_BIAS_FACTOR: float
    FP16: bool

    SCHEDULER_NAME: str
    LR_SCHEDULER: str
    LR_KWARGS: Dict[str, str]

    STEP_VERBOSE: int

    def __init__(self, **kwargs):

        self.OPTIMIZER_BUILDER = kwargs.get(
            "OPTIMIZER_BUILDER", "ClassificationOptimizer"
        )
        self.MODEL_SERVING = kwargs.get("MODEL_SERVING", False)
        self.EPOCHS = kwargs.get("EPOCHS", 10)
        self.SKIPEVAL = kwargs.get("SKIPEVAL", False)
        self.TEST_FREQUENCY = kwargs.get("TEST_FREQUENCY", 5)
        self.TRAINER = kwargs.get("TRAINER", "ClassificationTrainer")

        self.MODEL_VERSION = kwargs.get("MODEL_VERSION", 1)
        self.MODEL_CORE_NAME = kwargs.get("MODEL_CORE_NAME", "model")
        self.MODEL_BACKBONE = kwargs.get("MODEL_BACKBONE", "res18")
        self.MODEL_QUALIFIER = kwargs.get("MODEL_QUALIFIER", "all")
        self.DRIVE_BACKUP = kwargs.get("DRIVE_BACKUP", False)
        self.SAVE_FREQUENCY = kwargs.get("SAVE_FREQUENCY", 5)
        self.CHECKPOINT_DIRECTORY = kwargs.get("CHECKPOINT_DIRECTORY", "checkpoint")

        self.SHAPE = kwargs.get("SHAPE", [100, 100])
        self.NORMALIZATION_MEAN = kwargs.get("NORMALIZATION_MEAN", 0.5)
        self.NORMALIZATION_STD = kwargs.get("NORMALIZATION_STD", 0.5)
        self.NORMALIZATION_SCALE = kwargs.get("NORMALIZATION_SCALE", 255)
        self.H_FLIP = kwargs.get("H_FLIP", 0)
        self.T_CROP = kwargs.get("T_CROP", False)
        self.RANDOM_ERASE = kwargs.get("RANDOM_ERASE", False)
        self.RANDOM_ERASE_VALUE = kwargs.get("RANDOM_ERASE_VALUE", 0.5)
        self.CHANNELS = kwargs.get("CHANNELS", 3)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 32)
        self.WORKERS = kwargs.get("WORKERS", 2)

        self.BUILDER = kwargs.get("BUILDER", "classification_model_builder")
        self.MODEL_ARCH = kwargs.get("MODEL_ARCH", "ClassificationResnet")
        self.MODEL_BASE = kwargs.get("MODEL_BASE", "resnet18")
        self.MODEL_NORMALIZATION = kwargs.get("MODEL_NORMALIZATION", "bn")
        self.MODEL_KWARGS = kwargs.get("MODEL_KWARGS", {})
        self.PARAMETER_GROUPS = kwargs.get("PARAMETER_GROUPS", ['opt-1'])

        self.LOSSES = kwargs.get("LOSSES", [])
        self.LOSS_KWARGS = kwargs.get("LOSS_KWARGS", [])
        self.LAMBDAS = kwargs.get("LAMBDAS", [1.0])
        self.LOSS_LABEL = kwargs.get("LOSS_LABEL", "color")
        self.LOSS_NAME = kwargs.get("LOSS_NAME", "loss-1")

        self.OPTIMIZER_NAME = kwargs.get("OPTIMIZER_NAME", "opt-1")
        self.OPTIMIZER = kwargs.get("OPTIMIZER", "Adam")
        self.OPTIMIZER_KWARGS = kwargs.get("OPTIMIZER_KWARGS", {})
        self.BASE_LR = kwargs.get("BASE_LR", 0.001)
        self.LR_BIAS_FACTOR = kwargs.get("LR_BIAS_FACTOR", 1.0)
        self.WEIGHT_DECAY = kwargs.get("WEIGHT_DECAY", 0.0005)
        self.WEIGHT_BIAS_FACTOR = kwargs.get("WEIGHT_BIAS_FACTOR", 0.0005)
        self.FP16 = kwargs.get("FP16", False)

        self.SCHEDULER_NAME = kwargs.get("SCHEDULER_NAME", "opt-1")
        self.LR_SCHEDULER = kwargs.get("LR_SCHEDULER", "StepLR")
        self.LR_KWARGS = kwargs.get("LR_KWARGS", {"step_size": 20})

        self.STEP_VERBOSE = kwargs.get("STEP_VERBOSE", 100)