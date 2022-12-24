from typing import List, Dict, Tuple, Type
import os, builtins, json, yaml
import logging
import sys
from ednaml.exceptions import ErrorDuringImport
import importlib.util
import enum


class StorageArtifactType(enum.Enum):
    MODEL = "model"
    LOG = "log"
    CONFIG = "config"
    PLUGIN = "plugin"
    METRIC = "metric"
    ARTIFACT = "artifact"
    CODE = "code"
    EXTRAS= "extras"


class ExperimentKey:
    model_core_name: str
    model_version: str
    model_backbone: str
    model_qualifier: str

    def __init__(self, model_core_name, model_version, model_backbone, model_qualifier):
        self.model_backbone = str(model_backbone)
        self.model_version = str(model_version)
        self.model_core_name = str(model_core_name)
        self.model_qualifier = str(model_qualifier)

    def getKey(self):
        return (
            self.model_core_name,
            self.model_version,
            self.model_backbone,
            self.model_qualifier,
        )

    def getExperimentName(self):
        return "_".join(
            [
                self.model_core_name,
                self.model_version,
                self.model_backbone,
                self.model_qualifier,
            ]
        )

    def __repr__(self) -> str:
        dicts = json.dumps(self, default=config_serializer)  # or getvars()????
        dicta = json.loads(dicts)
        return yaml.dump(dicta)

    def __str__(self) -> str:
        return self.getExperimentName()

    def todict(self):
        return {
            "model_core_name": self.model_core_name,
            "model_version": self.model_version,
            "model_backbone": self.model_backbone,
            "model_qualifier": self.model_backbone,
        }


class RunKey:
    run: int

    def __init__(self, run):
        self.run = run

    def __repr__(self) -> str:
        dicts = json.dumps(self, default=lambda o: o.__dict__)  # or getvars()????
        dicta = json.loads(dicts)
        return yaml.dump(dicta)

    def getRunKey(self) -> str:
        return str(self.run)

    def __str__(self) -> str:
        return self.getRunKey()


class StorageKey:
    epoch: int
    step: int
    artifact: StorageArtifactType

    def __init__(self, epoch, step, artifact):
        self.epoch = epoch
        self.step = step
        self.artifact = artifact

    def __repr__(self) -> str:
        dicts = json.dumps(self.__dict__, default=lambda o: o.value)  # or getvars()????
        dicta = json.loads(dicts)
        return yaml.dump(dicta)

    def getStorageKey(self) -> str:
        return "-".join([self.artifact.value, str(self.epoch), str(self.step)])

    def __str__(self) -> str:
        return self.getStorageKey()


class ERSKey:
    experiment: ExperimentKey
    run: RunKey
    storage: StorageKey

    def __init__(self, experiment: ExperimentKey, run: RunKey, storage: StorageKey):
        self.experiment = experiment
        self.run = run
        self.storage = storage

    def __repr__(self) -> str:
        return "\n".join([repr(self.experiment), repr(self.run), repr(self.storage)])

    def printKey(self) -> str:
        return (
            "<"
            + ", ".join(
                [str(item) for item in [self.experiment, self.run, self.storage]]
            )
            + ">"
        )


class KeyMethods:
    @staticmethod
    def cloneExperimentKey(experiment_key: ExperimentKey) -> ExperimentKey:
        return ExperimentKey(
            model_core_name=experiment_key.model_core_name,
            model_backbone=experiment_key.model_backbone,
            model_qualifier=experiment_key.model_qualifier,
            model_version=experiment_key.model_version,
        )

    @staticmethod
    def cloneRunKey(run_key: RunKey):
        return RunKey(run=run_key.run)

    @staticmethod
    def cloneStorageKey(storage_key: StorageKey):
        return StorageKey(
            epoch=storage_key.epoch,
            step=storage_key.step,
            artifact=storage_key.artifact,
        )

    @staticmethod
    def cloneERSKey(ers_key: ERSKey):
        return ERSKey(
            experiment=KeyMethods.cloneExperimentKey(ers_key.experiment),
            run=KeyMethods.cloneRunKey(ers_key.run),
            storage=KeyMethods.cloneStorageKey(ers_key.storage),
        )



def locate_class(
    package="ednaml",
    subpackage="core",
    classpackage=None,
    classfile=None,
    forceload=0,
):
    """Locate an object by name or dotted path, importing as necessary."""
    if classpackage is None:
        if subpackage is not None:
            parts = package.split(".") + [subpackage]
        else:
            parts = package.split(".")
    else:
        if classfile is None:
            parts = package.split(".") + [subpackage, classpackage]
        else:
            parts = package.split(".") + [subpackage, classfile, classpackage]
    module, n = None, 0
    while n < len(parts):
        nextmodule = safeimport(".".join(parts[: n + 1]), forceload)
        if nextmodule:
            module, n = nextmodule, n + 1
        else:
            break
    if module:
        object = module
    else:
        object = builtins
    for part in parts[n:]:
        try:
            object = getattr(object, part)
        except AttributeError:
            return None
    return object


def safeimport(path, forceload=0, cache={}):
    """Import a module; handle errors; return None if the module isn't found.
    If the module *is* found but an exception occurs, it's wrapped in an
    ErrorDuringImport exception and reraised.  Unlike __import__, if a
    package path is specified, the module at the end of the path is returned,
    not the package at the beginning.  If the optional 'forceload' argument
    is 1, we reload the module from disk (unless it's a dynamic extension)."""
    try:
        # If forceload is 1 and the module has been previously loaded from
        # disk, we always have to reload the module.  Checking the file's
        # mtime isn't good enough (e.g. the module could contain a class
        # that inherits from another module that has changed).
        if forceload and path in sys.modules:
            if path not in sys.builtin_module_names:
                # Remove the module from sys.modules and re-import to try
                # and avoid problems with partially loaded modules.
                # Also remove any submodules because they won't appear
                # in the newly loaded module's namespace if they're already
                # in sys.modules.
                subs = [m for m in sys.modules if m.startswith(path + ".")]
                for key in [path] + subs:
                    # Prevent garbage collection.
                    cache[key] = sys.modules[key]
                    del sys.modules[key]
        module = __import__(path)
    except:
        # Did the error occur before or after the module was found?
        (exc, value, tb) = info = sys.exc_info()
        if path in sys.modules:
            # An error occurred while executing the imported module.
            raise ErrorDuringImport(sys.modules[path].__file__, info)
        elif exc is SyntaxError:
            # A SyntaxError occurred before we could execute the module.
            raise ErrorDuringImport(value.filename, info)
        elif issubclass(exc, ImportError) and value.name == path:
            # No such module in the path.
            return None
        else:
            # Some other error occurred during the importing process.
            raise ErrorDuringImport(path, sys.exc_info())
    for part in path.split(".")[1:]:
        try:
            module = getattr(module, part)
        except AttributeError:
            return None
    return module


def path_import(absolute_path):
    """implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly"""
    """And from https://stackoverflow.com/questions/41861427/python-3-5-how-to-dynamically-import-a-module-given-the-full-file-path-in-the"""
    spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extend_mean_arguments(
    params_to_fix: List[str] = [0.5, 0.5], channels=3
) -> Tuple[List[float]]:
    return tuple([[item] * channels for item in params_to_fix])


def config_serializer(obj):
    return obj.__dict__


def generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME):
    logger = logging.getLogger(MODEL_SAVE_FOLDER)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    logger_save_path = os.path.join(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    fh = logging.FileHandler(logger_save_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s-%(msecs)d %(message)s", datefmt="%H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    cs.setFormatter(
        logging.Formatter("%(asctime)s-%(msecs)d %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(cs)
    return logger


# TODO handle tuple return
def generate_save_names(cfg):
    MODEL_SAVE_NAME = "%s-v%i" % (
        cfg.get("SAVE.MODEL_CORE_NAME"),
        cfg.get("SAVE.MODEL_VERSION"),
    )
    MODEL_SAVE_FOLDER = "%s-v%i-%s-%s" % (
        cfg.get("SAVE.MODEL_CORE_NAME"),
        cfg.get("SAVE.MODEL_VERSION"),
        cfg.get("SAVE.MODEL_BACKBONE"),
        cfg.get("SAVE.MODEL_QUALIFIER"),
    )
    LOGGER_SAVE_NAME = "%s-v%i-%s-%s-logger.log" % (
        cfg.get("SAVE.MODEL_CORE_NAME"),
        cfg.get("SAVE.MODEL_VERSION"),
        cfg.get("SAVE.MODEL_BACKBONE"),
        cfg.get("SAVE.MODEL_QUALIFIER"),
    )
    return (
        MODEL_SAVE_NAME,
        MODEL_SAVE_FOLDER,
        LOGGER_SAVE_NAME,
        "",
    )


def fix_generator_arguments(
    cfg: Dict, params_to_fix: List[str] = []
) -> Tuple[List[float]]:
    if len(params_to_fix) > 0:
        return_params = [None] * len(params_to_fix)
        for idx, param in enumerate(params_to_fix):
            if type(cfg.get(param)) is int or type(cfg.get(param)) is float:
                return_params[idx] = [cfg.get(param)] * cfg.get(
                    "TRANSFORMATION.CHANNELS"
                )
        return tuple(return_params)
    else:
        if (
            type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is int
            or type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is float
        ):
            NORMALIZATION_MEAN = [
                cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")
            ] * cfg.get("TRANSFORMATION.CHANNELS")
        if (
            type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is int
            or type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is float
        ):
            NORMALIZATION_STD = [cfg.get("TRANSFORMATION.NORMALIZATION_STD")] * cfg.get(
                "TRANSFORMATION.CHANNELS"
            )
        if (
            type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is int
            or type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is float
        ):
            RANDOM_ERASE_VALUE = [
                cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")
            ] * cfg.get("TRANSFORMATION.CHANNELS")
        return NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE


model_weights = {
    "resnet18": [
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet18-5c106cde.pth",
    ],
    "resnet34": [
        "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "resnet34-333f7ec4.pth",
    ],
    "resnet50": [
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "resnet50-19c8e357.pth",
    ],
    "resnet101": [
        "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "resnet101-5d3b4d8f.pth",
    ],
    "resnet152": [
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
        "resnet152-b121ed2d.pth",
    ],
    "resnext50_32x4d": [
        "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "resnext50_32x4d-7cdf4587.pth",
    ],
    "resnext101_32x8d": [
        "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "resnext101_32x8d-8ba56ff5.pth",
    ],
    "wide_resnet50_2": [
        "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        "wide_resnet50_2-95faca4d.pth",
    ],
    "wide_resnet101_2": [
        "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        "wide_resnet50_2-95faca4d.pth",
    ],
    "resnet18_cbam": [
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet18-5c106cde_cbam.pth",
    ],
    "resnet34_cbam": [
        "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "resnet34-333f7ec4_cbam.pth",
    ],
    "resnet50_cbam": [
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "resnet50-19c8e357_cbam.pth",
    ],
    "resnet101_cbam": [
        "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "resnet101-5d3b4d8f_cbam.pth",
    ],
    "resnet152_cbam": [
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
        "resnet152-b121ed2d_cbam.pth",
    ],
    "shufflenetv2_small": [
        "https://github.com/asuprem/shufflenet-models/raw/master/ShuffleNetV2%2B.Small.pth.tar",
        "shufflenetv2-small.pth",
    ],
}

from ednaml.utils.LabelMetadata import LabelMetadata


def merge_dictionary_on_key_with_copy(a, b, path=None):
    from copy import deepcopy

    return merge_dictionary_on_key(deepcopy(a), b)


def merge_dictionary_on_key(a, b, path=None):  # make it more descriptive!!!
    # from https://stackoverflow.com/a/7205107/2601357
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dictionary_on_key(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                a[key]
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

# https://stackoverflow.com/a/40389411/2601357
def dict_to_table(myDict, colList=None):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList: colList = list(myDict[0].keys() if myDict else [])
    myList = [colList] # 1st row = header
    for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
    colSize = [max(map(len,col)) for col in zip(*myList)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    myList.insert(1, ['-' * i for i in colSize]) # Seperating line
    return "\n".join([formatStr.format(*item) for item in myList])
