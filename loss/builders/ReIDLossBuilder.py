from . import LossBuilder

from ..SoftmaxLogitsLoss import SoftmaxLogitsLoss
from ..SoftmaxLabelSmooth import SoftmaxLabelSmooth
from ..TripletLoss import TripletLoss
from ..MarginLoss import MarginLoss
from ..ContrastiveLoss import ContrastiveLoss
from ..CompactContrastiveLoss import CompactContrastiveLoss

class ReIDLossBuilder(LossBuilder):
  LOSS_PARAMS = {}
  LOSS_PARAMS['SoftmaxLogitsLoss'] = {}
  LOSS_PARAMS['SoftmaxLogitsLoss']['fn'] = SoftmaxLogitsLoss
  LOSS_PARAMS['SoftmaxLogitsLoss']['args'] = ['logits', 'labels']
  LOSS_PARAMS['TripletLoss'] = {}
  LOSS_PARAMS['TripletLoss']['fn'] = TripletLoss
  LOSS_PARAMS['TripletLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['MarginLoss'] = {}
  LOSS_PARAMS['MarginLoss']['fn'] = MarginLoss
  LOSS_PARAMS['MarginLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['SoftmaxLabelSmooth'] = {}
  LOSS_PARAMS['SoftmaxLabelSmooth']['fn'] = SoftmaxLabelSmooth
  LOSS_PARAMS['SoftmaxLabelSmooth']['args'] = ['logits', 'labels']
  LOSS_PARAMS['ContrastiveLoss'] = {}
  LOSS_PARAMS['ContrastiveLoss']['fn'] = ContrastiveLoss
  LOSS_PARAMS['ContrastiveLoss']['args'] = ['features', 'labels']
  LOSS_PARAMS['CompactContrastiveLoss'] = {}
  LOSS_PARAMS['CompactContrastiveLoss']['fn'] = CompactContrastiveLoss
  LOSS_PARAMS['CompactContrastiveLoss']['args'] = ['features', 'labels', 'epoch']

  def __init__(self, loss_functions, loss_lambda, loss_kwargs, **kwargs):
    self.loss = []
        
    fn_len = len(loss_functions)
    lambda_len = len(loss_lambda)
    kwargs_len = len(loss_kwargs)
    # Set up the logger
    self.logger = kwargs.get("logger")
    # Sanity check
    if fn_len != lambda_len:
      raise ValueError("Loss function list length is %i. Expected %i length loss_lambdas, got %i"%(fn_len, fn_len, lambda_len))
    if fn_len != kwargs_len:
      raise ValueError("Loss function list length is %i. Expected %i length loss_kwargs, got %i"%(fn_len, fn_len, kwargs_len))
    # Add the loss functions with the correct features. Lambda is applied later
    for idx, fn in enumerate(loss_functions):
      self.loss.append(self.LOSS_PARAMS[fn]['fn'](**loss_kwargs[idx]))
      self.logger.info("Added {loss} with lambda = {lamb} and loss arguments {largs}".format(loss=fn, lamb=loss_lambda[idx], largs=str(loss_kwargs[idx])))
    
    lambda_sum = sum(loss_lambda)
    loss_lambda = [float(item)/float(lambda_sum) for item in loss_lambda]
    self.loss_lambda = loss_lambda
    self.loss_fn = loss_functions