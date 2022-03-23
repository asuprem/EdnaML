import pdb
from pydoc import classname
from torch import nn, softmax
from models.ClassificationResnet import ClassificationResnet
from utils import layers
import torch


class MultiClassificationResnet(ClassificationResnet):
    """Multiclassification Resnet model, that performs multiple classifications from the same backbone.

    A MulticlassificationResnet model is a base ResNet with multiple FC classification layers.

    Args: (TODO)
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        metadata (Dict[str:int], None): FC dimensions for each classification, keyed by class names

    Kwargs (MODEL_KWARGS):
        number_outputs (int): Number of different FC layers connected to the feature layer of the backbone.
        softmax_dimensions (List[int]): Classes of each FC layer, in order. Optional. If not provided, `MultiClassificationResnet` will use `output_classnames` and `metadata` to infer dimension size
        output_classnames (List[str]): The name for each output, in order. These should be the same as the label names for the multiple classes.
        labelnames (List[str]): The order of labels provided by the crawler. This is used during model training, where crawler ground truth labels must be matched to model outputs. 
        
        last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
        attention (str, None): The attention module to use. Only supports ['cbam', 'dbam']
        input_attention (bool, false): Whether to include the IA module
        secondary_attention (List[int], None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic blocks provided here in List.
        part_attention (bool): Whether to use local attention

    Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS):
        zero_init_residual (bool, false): Whether the final layer uses zero initialization
        top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        num_classes (int, 1000): Number of features in final imagenet FC layer
        groups (int, 1): Used during resnet variants construction
        width_per_group (int, 64): Used during resnet variants construction
        replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
        norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D

    Methods: 
        forward: Process a batch

    """

    def __init__(self, base = 'resnet50', weights=None, normalization=None, metadata=None, **kwargs):
        """We will inherit the base construction from ClassificationResNet, and modify the softmax head.

        Args:
            base (str, optional): _description_. Defaults to 'resnet50'.
            weights (_type_, optional): _description_. Defaults to None.
            normalization (_type_, optional): _description_. Defaults to None.
            metadata (_type_, optional): _description_. Defaults to None.
        """
        self.metadata = metadata
        self.number_outputs = kwargs.get("number_outputs", 1)
        self.softmax_dimensions = kwargs.get("softmax_dimensions", None)
        self.output_classnames = kwargs.get("output_classnames", None)
        self.labelnames = kwargs.get("labelnames", None)
        
        super().__init__(base, weights, normalization, **kwargs)
        

    def build_softmax(self, **kwargs):
        """Build the softmax layers, using info either in self.softmax_dimensions or by combining metadata info of labelname->numclasses and the outputclassnames
        """
        
        if self.softmax_dimensions is None:
            # sets the size of softmax_dimensions to match number of outputs in this model...
            self.softmax_dimensions = [None]*(self.number_outputs)

            # TODO we assume if softmax_dimensions is none, that the output_classnames is constructed properly. Handle the bad case, by passing in a logger instance to this to log warnings and errors
            # Also, need to check errors where if we are inferring softmax_dimensions, there should only be 1 output, and if numberoutputs>1 while output_classnames is not provided, throw an error, etc.
            # Then later, adjust this for re-id case where there may be no softmax, and direct inference from features.
            if self.output_classnames is None and self.number_outputs > 1:
                raise ValueError("`number_outputs`>1, but no output dimensions or class names provided. If you want automatic inference of number of softmax dimensions, provide ")
            elif self.output_classnames is None and self.number_outputs == 1:
                # Here we infer using metadata
                self.softmax_dimensions[0] = self.metadata[self.metadata.keys()[0]]
                self.output_classnames = [self.metadata.keys()[0]]
            elif self.output_classnames is not None:
                if len(self.output_classnames)!=self.number_outputs:
                    raise ValueError("Length of output_classnames MUST match number_outputs. Expected %i and got %i"%(self.number_outputs, len(self.output_classnames)))
                for idx,classname in enumerate(self.output_classnames):
                    self.softmax_dimensions[idx] = self.metadata[classname] # This will raise KeyError if the classname is not in metadata...
            else:
                raise RuntimeError("Case of number_outputs ", self.number_outputs, " with output_classnames ", self.output_classnames, " not handled. Oops.")

        # NOTE, for re-id type models...multiclassification model will anyway yield the features with softmax outputs, so we don't have to worry about that...
        # For pure-reid model, probably best to use ClassificationResNet and modify to use no softmax...TODO this is a future step...
        tsoftmax = [None]*self.number_outputs
        for idx,fc_dimension in enumerate(self.softmax_dimensions):
            tsoftmax[idx] = nn.Linear(self.embedding_dimensions, fc_dimension, bias=False)
            tsoftmax[idx].apply(self.weights_init_softmax)
        self.softmax = nn.ModuleList(tsoftmax)

    def base_forward(self,x):
        features = self.gap(self.base(x))
        features = features.view(features.shape[0],-1)
        features = self.emb_linear(features)
        return features


    def forward_impl(self,x):
        features = self.base_forward(x)
        
        #if self.feat_norm is not None: <-- no need, identity
        features = self.feat_norm(features)

        softmax_logits = [None]*self.number_outputs
        for idx,softmaxlayer in enumerate(self.softmax):
            softmax_logits[idx] = softmaxlayer(features)
        return softmax_logits, features, []   # soft logits are the softmax logits we will use to for training. We can use features to store the historical probability????
  
    