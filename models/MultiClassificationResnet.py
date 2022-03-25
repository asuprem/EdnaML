import pdb
from pydoc import classname
from torch import nn, softmax
from models.ClassificationResnet import ClassificationResnet
from utils import layers
import torch

from utils.LabelMetadata import LabelMetadata


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

    model_name = "MultiClassificationResNet"
    model_arch = "MultiClassificationResNet"
    number_outputs = 1
    output_names = ["out0"]
    softmax_dimensions = [None]
    output_labels = ["color"]
    secondary_outputs = []

    _internal_name_count=0

    def __init__(self, base = 'resnet50', weights=None, normalization=None, metadata=None, **kwargs):
        """We will inherit the base construction from ClassificationResNet, and modify the softmax head.

        Args:
            base (str, optional): _description_. Defaults to 'resnet50'.
            weights (_type_, optional): _description_. Defaults to None.
            normalization (_type_, optional): _description_. Defaults to None.
            metadata (_type_, optional): _description_. Defaults to None.
        """
        
        
        super().__init__(base=base, weights=weights, normalization=normalization, metadata=metadata, **kwargs)
        

    def model_attributes_setup(self, **kwargs):

        self.embedding_dimensions = kwargs.get("embedding_dimensions", None)
        if self.normalization == '':
            self.normalization = None
        
        

        self.number_outputs = kwargs.get("number_outputs", 1)

        outputs = kwargs.get("outputs", [{  "name":self._internal_name_counter(), 
                                            "label":"color",
                                            "dimensions":None }] )

        if len(outputs)!=self.number_outputs:
            raise ValueError("Mismatch in length of outputs %i and number of outputs %i"%(len(outputs), self.number_outputs))
        
        self.softmax_dimensions = [None]*self.number_outputs
        self.output_names = [None]*self.number_outputs
        self.output_labels = [None]*self.number_outputs

        for idx, output_details in enumerate(outputs):
            self.softmax_dimensions[idx] = output_details.get("dimensions", None)
            self.output_names[idx] = output_details.get("name", self._internal_name_counter())
            self.output_labels[idx] = output_details["label"]

        self.base = None
        self.gap = None
        self.emb_linear = None
        self.feat_norm = None
        self.softmax = None

    def _internal_name_counter(self):
        out="out"+str(self._internal_name_count)
        self._internal_name_count+=1
        return out

    def build_softmax(self, **kwargs):
        """Build the softmax layers, using info either in self.softmax_dimensions or by combining metadata info of labelname->numclasses and the outputclassnames
        """
        for idx in range(self.number_outputs):
            if self.softmax_dimensions[idx] is None:
                # TODO we assume if softmax_dimensions is none, that the output_classnames is constructed properly. Handle the bad case, by passing in a logger instance to this to log warnings and errors
                if self.output_labels[idx] is None:
                    raise ValueError("No label provided for output %i. Cannot automatically infer dimensions"%idx)
                self.softmax_dimensions[idx] = self.metadata.getLabelDimensions(self.output_labels[idx])

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


    def forward_impl(self,x, **kwargs):
        features = self.base_forward(x)
        
        #if self.feat_norm is not None: <-- no need, identity
        features = self.feat_norm(features)

        softmax_logits = [None]*self.number_outputs
        for idx,softmaxlayer in enumerate(self.softmax):
            softmax_logits[idx] = softmaxlayer(features)
        return softmax_logits, features, []   # soft logits are the softmax logits we will use to for training. We can use features to store the historical probability????
  
    