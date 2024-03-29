from typing import Dict, List

from ednaml.config import BaseConfig


class ConfigDefaults(BaseConfig):
    OPTIMIZER_BUILDER: str
    MODEL_SERVING: bool
    EPOCHS: int
    SKIPEVAL: bool
    TEST_FREQUENCY: int
    TRAINER: str
    TRAINER_ARGS: Dict[str, str]
    MODEL_VERSION: int
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    DRIVE_BACKUP: bool
    LOG_BACKUP: bool
    SAVE_FREQUENCY: int
    STEP_SAVE_FREQUENCY: int
    CHECKPOINT_DIRECTORY: str

    OUTPUT_ARGS: Dict[str,str]
    DEPLOY: str
    DEPLOYMENT_ARGS: Dict[str,str]
    DEPLOYMENT_EPOCHS: int

    TRANSFORM_ARGS: Dict
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
    INPUT_SIZE: List[int]
    PLUGIN: str
    PLUGIN_NAME: str
    PLUGIN_KWARGS: Dict[str, str]

    STORAGE_TYPE: str
    STORAGE_ARGS: Dict
    STORAGE_URL: str

    def __init__(self, **kwargs):

        self.OPTIMIZER_BUILDER = kwargs.get(
            "OPTIMIZER_BUILDER", "ClassificationOptimizer"
        )
        self.MODEL_SERVING = kwargs.get("MODEL_SERVING", False)
        self.EPOCHS = kwargs.get("EPOCHS", 10)
        self.SKIPEVAL = kwargs.get("SKIPEVAL", False)
        self.TEST_FREQUENCY = kwargs.get("TEST_FREQUENCY", 1)
        self.TRAINER = kwargs.get("TRAINER", "BaseTrainer")
        self.TRAINER_ARGS = kwargs.get(
            "TRAINER_ARGS",
            {
                "accumulation_steps": 1,
            },
        )

        self.MODEL_VERSION = kwargs.get("MODEL_VERSION", 1)
        self.MODEL_CORE_NAME = kwargs.get("MODEL_CORE_NAME", "model")
        self.MODEL_BACKBONE = kwargs.get("MODEL_BACKBONE", "backbone")
        self.MODEL_QUALIFIER = kwargs.get("MODEL_QUALIFIER", "all")
        self.DRIVE_BACKUP = kwargs.get("DRIVE_BACKUP", False)
        self.LOG_BACKUP = kwargs.get("LOG_BACKUP", False)
        self.SAVE_FREQUENCY = kwargs.get("SAVE_FREQUENCY", 5)
        self.STEP_SAVE_FREQUENCY = kwargs.get("STEP_SAVE_FREQUENCY", 0)
        self.CHECKPOINT_DIRECTORY = kwargs.get(
            "CHECKPOINT_DIRECTORY", "checkpoint"
        )

        self.OUTPUT_ARGS = {}
        self.DEPLOY = kwargs.get("DEPLOY", "BaseDeploy")
        self.DEPLOYMENT_ARGS = kwargs.get("DEPLOYMENT_ARGS", {})
        self.DEPLOYMENT_EPOCHS = kwargs.get("DEPLOYMENT_EPOCHS", 1)

        self.TRANSFORM_ARGS = {}
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 32)
        self.WORKERS = kwargs.get("WORKERS", 2)

        self.BUILDER = kwargs.get("BUILDER", "ednaml_model_builder")
        self.MODEL_ARCH = kwargs.get("MODEL_ARCH", "ModelAbstract")
        self.MODEL_BASE = kwargs.get("MODEL_BASE", "base")
        self.MODEL_NORMALIZATION = kwargs.get("MODEL_NORMALIZATION", "bn")
        self.MODEL_KWARGS = kwargs.get("MODEL_KWARGS", {})
        self.PARAMETER_GROUPS = kwargs.get("PARAMETER_GROUPS", ["opt-1"])

        self.LOSSES = kwargs.get("LOSSES", [])
        self.LOSS_KWARGS = kwargs.get("LOSS_KWARGS", [])
        self.LAMBDAS = kwargs.get("LAMBDAS", [])
        self.LOSS_LABEL = kwargs.get("LOSS_LABEL", "")
        self.LOSS_NAME = kwargs.get("LOSS_NAME", "")

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
        self.INPUT_SIZE = kwargs.get("INPUT_SIZE", None)

        self.STORAGE_TYPE = kwargs.get("STORAGE_TYPE", "BaseStorage")
        self.STORAGE_ARGS = kwargs.get("STORAGE_ARGS", {})
        self.STORAGE_URL = kwargs.get("STORAGE_URL", "./")
        
        self.PLUGIN_NAME = "mp-1"
        self.PLUGIN = "ModelPlugin"
        self.PLUGIN_KWARGS = {}
