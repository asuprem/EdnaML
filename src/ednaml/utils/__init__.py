from typing import List, Dict, Tuple
import os, builtins
import logging
import sys
from ednaml.exceptions import ErrorDuringImport


def locate_class(package="ednaml",subpackage="core", classpackage="EdnaML", classfile=None, forceload=0):
    """Locate an object by name or dotted path, importing as necessary."""
    if classfile is None:
        parts = package.split(".")+[subpackage, classpackage]
    else:
        parts = package.split(".")+ [subpackage, classfile, classpackage]
    module, n = None, 0
    while n < len(parts):
        nextmodule = safeimport('.'.join(parts[:n+1]), forceload)
        if nextmodule: module, n = nextmodule, n + 1
        else: break
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
                subs = [m for m in sys.modules if m.startswith(path + '.')]
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
    for part in path.split('.')[1:]:
        try: module = getattr(module, part)
        except AttributeError: return None
    return module

def dynamic_import(cfg, module_name: str, import_name: str, default: str = None):
    """ Perform a dynamic import 

    Args:
        cfg (kaptan Kaptan object): A kaptan configuration object
        module_name (str): The directory where the module-to-import resides. For example to import a crawler, module_name should be 'crawlers'
        import_name (str): The key in the kaptan configuration that provides the import name
        default (str): If import_name does not exist, `default` will be used instead

    Returns:
        Imported module attribute

    Raises:
        ModuleNotFoundError (implicit): If provided module is not found
        AttributeError (implicit): If provided import_name or default does not exist
    """

    import_name = cfg.get(import_name, default)
    imported_module = __import__(
        "%s." % module_name + import_name, fromlist=[import_name]
    )
    return getattr(imported_module, import_name)


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
    if cfg.get("SAVE.DRIVE_BACKUP"):
        CHECKPOINT_DIRECTORY = (
            cfg.get("SAVE.CHECKPOINT_DIRECTORY", "./drive/My Drive/Vehicles/Models/")
            + MODEL_SAVE_FOLDER
        )
    else:
        CHECKPOINT_DIRECTORY = ""
    return MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY


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
