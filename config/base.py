from yacs.config import CfgNode as CN 

_C = CN()

_C.DATASET = CN()
_C.DATASET.ROOT_DATA_FOLDER = "VeRi"
_C.DATASET.TRAIN_FOLDER = "image_train"
_C.DATASET.TEST_FOLDER = "image_test"
_C.DATASET.QUERY_FOLDER = "image_query"
_C.DATASET.SHAPE = (208,208)

_C.TRANSFORMATION = CN()
_C.TRANSFORMATION.NORMALIZATION_MEAN = 0.5
_C.TRANSFORMATION.NORMALIZATION_STD = 0.5
_C.TRANSFORMATION.NORMALIZATION_SCALE = 1./255.
_C.TRANSFORMATION.H_FLIP = 0.5
_C.TRANSFORMATION.T_CROP = True
_C.TRANSFORMATION.RANDOM_ERASE = True
_C.TRANSFORMATION.RANDOM_ERASE_VALUE = 0.5
_C.TRANSFORMATION.CHANNELS = 3
_C.TRANSFORMATION.BATCH_SIZE = 36
_C.TRANSFORMATION.INSTANCES = 6
_C.TRANSFORMATION.WORKERS = 8

_C.MODEL = CN()
_C.MODEL.MODEL_BASE = 'resnet50'
_C.MODEL.MODEL_WEIGHTS = True
_C.MODEL.MODEL_ATTENTION = ''
_C.MODEL.EMB_DIM = 2048
_C.MODEL.MODEL_NORMALIZATION = 'bn'

_C.SAVE = CN()
_C.SAVE.SAVE_FREQUENCY = 5 # Epoch
_C.SAVE.MODEL_VERSION = 1
_C.SAVE.MODEL_CORE_NAME = "torch_reid"
_C.SAVE.MODEL_BACKBONE = "res50"
_C.SAVE.MODEL_QUALIFIER = "all"
_C.SAVE.DRIVE_BACKUP = True

_C.EXECUTION = CN()
_C.EXECUTION.MODE = "TRAIN"
_C.EXECUTION.MODEL_SERVING = ''
_C.EXECUTION.EPOCHS = 260
_C.EXECUTION.TEST_FREQUENCY = 5

_C.LOSS = CN()
_C.LOSS.LOSSES = ['SoftmaxLogitsLoss', 'TripletLoss']
_C.LOSS.LOSS_KWARGS = ["{}", "{'margin':0.3, 'mine':'hard'}"]
_C.LOSS.LOSS_LAMBDAS = [1.0, 1.0]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER_NAME = "Adam"
_C.OPTIMIZER.OPTIMIZER_KWARGS = "{}"
_C.OPTIMIZER.BASE_LR = 0.0001
_C.OPTIMIZER.LR_BIAS_FACTOR = 1.0
_C.OPTIMIZER.WEIGHT_DECAY = 0.0005
_C.OPTIMIZER.WEIGHT_BIAS_FACTOR = 0.0005
_C.OPTIMIZER.FP16 = True # Epoch

_C.SCHEDULER = CN()
_C.SCHEDULER.LR_SCHEDULER = 'FineGrainedSteppedLR'
_C.SCHEDULER.LR_KWARGS = "{'lr_ops':[(2,'+',2e-5), (4,'+',2e-5), (6,'+',2e-5), (8,'+',2e-5), (10,'+',2e-5), (20,'*',0.9), (40,'+',0.8), (60,'+',0.7), (80,'+',0.7), (100,'+',0.6), (120,'+',0.5)]}"

_C.LOGGING = CN()
_C.LOGGING.STEP_VERBOSE = 100 # Batch

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`