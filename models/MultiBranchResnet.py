import pdb
from pydoc import classname
from torch import nn, softmax
from backbones.multibranchresnet import multibranchresnet
from models.abstracts import ModelAbstract
from utils import layers
import torch

from utils.LabelMetadata import LabelMetadata


class MultiBranchResnet(ModelAbstract):
    """Multibranch Resnet model, that performs multiple classifications with different branches. Branches may be fused as well.

    A MultiBranchResnet model is a base ResNet with multiple FC classification layers.

    Args: (TODO)
        base (str): The architecture base for resnet, i.e. resnet50, resnet18
        weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
        normalization (str, None): Can be None, where it is torch's normalization. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
        metadata (Dict[str:int], None): FC dimensions for each classification, keyed by class names

    Kwargs (MODEL_KWARGS):
        - `number_branches`: This is the number of branches for the model
        - `branches`: A list, each element is the i-th branch's metadata
            - `name`: This is the name of this branch. This is used by model_builder to keep track of output names and labels. 
            - `number_outputs`: This is the number of classification outputs for this branch
            - `outputs`: A list, each element is the j-th output's metadata for this branch
                - `dimensions`: This is the number of classes for this output. This can be left blank if you want EdnaML to infer the size from the `LABEL` parameter below. 
                - `name`: This is the name of this output. 
                - `label`: This is the name of the label this output is tracking. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span> labels.
        - `fuse`: **Bool**. Whether branch outputs are going to be fused
        - `fuse_outputs`: List of output names (not branch names) that are fused
        - `fuse_dimensions`: The dimensions of the fused output. If left blank, EdnaML will infer from `fuse_label`
        - `fuse_label`: The label that tracks the fused output 
        
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

    model_name = "MultiBranchResNet"
    model_arch = "MultiBranchResNet"
    number_outputs = 1
    output_names = ["out0"]
    softmax_dimensions = [None]
    output_labels = ["color"]
    secondary_outputs = []
    base: multibranchresnet
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
        
        
        """
        When we do forward, we will receive outputs as:
        
            logits, features, secondary-outputs
        
        Each of logits, features, and secondary-outputs is a list. 
        
        The size of features corresponds to branch-features, +1 if they are fused.
        The size of logits, on the other hand, does not correspond to features,
        because each branch may have multiple outputs, ergo multiple logits. 
        
        Our order for outputs is determined by the order of branches. In future, we can also specify 
        in the configuration the order of outputs, but we will leave it alone for now.

        So, when doing forward, we will do, given input x:

        x = pre-branch(x)   <-- this is the bit before branching, i.e. for branching, we can adjust where it occurs in resnet, i.e. at which basic block...
        branch-feats = [none]*numbranches
        for branch in self.branches:        <-- in here, branches are in the order we specified in the config!!!
            branch_feats[idx] = self.branches[idx][base](x)
        if fused:
            fused-feats = self.fuse(branch-feats)       <-- at this point, we have all the features
        
        # Now, we will need to extract the outputs themselves
        outputs = [None]*self.num-outputs
        output_counter = 0
        for bidx, branch in self.branches:
            for softlayer in self.branches[idx]["softmax"]:       <-- this is a list of softmax within each branch, in order
                outputs[output_counter] = softlayer ( branch-feats[bidx] )
                output_counter += 1
        
        if fused:
            fused-outs = self.soft_fuse(fused-feats)

        return outputs+[fused-outs], branch_feats+[fused-feats], []



         [output1, output2, output3, ..., fuse-output], [feature1, feature2, fused-feature], []
        """
        
        # So, things we need
        # num branches
        # 
        branches = kwargs.get("branches")
        self.number_branches = kwargs.get("number_branches", 3)
        self.branches_meta = {branch["name"]:branch for branch in branches}
        self.branch_name_order = [branch["name"] for branch in branches]
        
        self.number_outputs = 0
        self.output_name_order = [] # order of outputs, by their name. We can look up specific metadata in outputs_meta,, such as label it is tracking
        self.outputs_meta = {}
        for branch_name in self.branch_name_order:
            nouts = self.branches_meta[branch_name]["number_outputs"]
            self.number_outputs+=nouts
            out_dict = {out["name"]:out for out in self.branches_meta[branch_name]["outputs"]}
            out_order = [out["name"] for out in self.branches_meta[branch_name]["outputs"]]
            self.output_name_order+=out_order

            self.outputs_meta = dict(self.outputs_meta, **out_dict)
            self.branches_meta[branch_name]["output_nameorder"] = [item for item in out_order]    # for list copy, instead of reference...this is the outputs for this branch
        # Now we adjust the dimensions, i.e. fix them if they do not exist
        for output in self.outputs_meta:
            if "dimensions" in self.outputs_meta[output]:
                if self.outputs_meta[output]["dimensions"] is None:
                    self.outputs_meta[output]["dimensions"] = self.metadata.getLabelDimensions(self.outputs_meta[output]["label"])
            else:
                self.outputs_meta[output]["dimensions"] = self.metadata.getLabelDimensions(self.outputs_meta[output]["label"])
        
        self.output_dimensions = []
        self.output_label_order = []
        for oname in self.output_name_order:
            self.output_dimensions+=[self.outputs_meta[oname]["dimensions"]]
            self.output_label_order+=[self.outputs_meta[oname]["label"]]

        # Now we have self.output_name_order, which contains output-names in order of how they will be output
        # Now we need to add the fused information
        self.branch_fuse = kwargs.get('fuse', False)
        self.branch_fuse_names = {item:1 for item in kwargs.get('fuse_outputs', [])}
        self.branch_fuse_idx = []
        if self.branch_fuse:
            self.branch_fuse_idx = [idx for idx,b_name in enumerate(self.branch_name_order) if b_name in self.branch_fuse_names]
        self.fuse_dimensions = kwargs.get('fuse_dimensions', None)
        self.fuse_label = kwargs.get('fuse_label', None)
        if self.fuse_dimensions is None:
            self.fuse_dimensions = self.metadata.getLabelDimensions(self.fuse_label)
        
        # need metadata for model_labelorder for the output
        self.model_labelorder = [item for item in self.output_label_order]
        if self.branch_fuse:
            self.model_labelorder += [self.fuse_label]
        
        self.base = None
        self.branches = {}
        for bname in self.branch_name_order:
            self.branches[bname] = {}
            self.branches[bname]["gap"] = None
            self.branches[bname]["emb_linear"] = None
            self.branches[bname]["feat_norm"] = None
            self.branches[bname]["softmax"] = [None]*self.branches_meta[bname]["number_outputs"]


    def model_setup(self, **kwargs):
        self.build_base(**kwargs)
        self.build_normalization()
        self.build_softmax()
        self.build_fused()

    def build_base(self, **kwargs):
        """Build the model base.

        Builds the architecture base/core.
        """
        _resnet = __import__("backbones.multibranchresnet", fromlist=["resnet"])
        _resnet = getattr(_resnet, self.model_base)
        # Set up the resnet backbone
        self.base = _resnet(last_stride=1, **kwargs)
        if self.weights is not None:
            #self.base.load_param(self.weights)
            pass
        
        if self.embedding_dimensions is None:
            self.embedding_dimensions = 512*self.base.block.expansion
        self.fused_feat_dimensions = len(self.branch_fuse_idx) * 512 * self.base.block.expansion

        for bname in self.branch_name_order:
            self.branches[bname]["gap"] = nn.AdaptiveAvgPool2d(1)
            self.branches[bname]["emb_linear"] = torch.nn.Identity()
            
            if self.embedding_dimensions != 512*self.base.block.expansion:
                 self.branches[bname]["emb_linear"] = nn.Linear(self.base.block.expansion*512, self.embedding_dimensions, bias=False)

            self.branches[bname]["feat_norm"] = None
            self.branches[bname]["softmax"] = [None]*self.branches_meta[bname]["number_outputs"]



    def build_normalization(self):

        norm_func=nn.Module
        norm_args={}
        norm_div=1
        if self.normalization == 'bn':
            norm_func = nn.BatchNorm1d
            norm_args={"affine":True}
        elif self.normalization == "in":
            norm_func = layers.FixedInstanceNorm1d
            norm_args={"affine":True}
        elif self.normalization == "gn":
            norm_div=16
            norm_func = nn.GroupNorm
            norm_args={"num_channels":self.embedding_dimensions, "affine":True}
        elif self.normalization == "ln":
            norm_func = nn.LayerNorm
            norm_args={"elementwise_affine":True}
        elif self.normalization == 'l2':
            norm_func = layers.L2Norm
            norm_args={"scale":1.0}            
        elif self.normalization is None or self.normalization == '':
            norm_func = torch.nn.Identity
            norm_args={}
        else:
            raise NotImplementedError()
        
        self.fused_feat_norm = norm_func(self.fused_feat_dimensions // norm_div, **norm_args)
        for bname in self.branch_name_order:
            self.branches[bname]["feat_norm"] = norm_func(self.embedding_dimensions // norm_div, **norm_args)


    def build_softmax(self):
        # We will build the softmax layers...
        # For this, we will go into the self.branches, and add the softmax there...
        for bname in self.branch_name_order:
            for idx,output_name in enumerate(self.branches_meta[bname]["output_nameorder"]):
                self.branches[bname]["softmax"][idx] = nn.Linear(   self.embedding_dimensions, 
                                                                    self.outputs_meta[output_name]["dimensions"], 
                                                                    bias=False)
                self.branches[bname]["softmax"][idx].apply(self.weights_init_softmax)
            self.branches[bname]["softmax"] = nn.ModuleList(self.branches[bname]["softmax"])

    def build_fused(self):
        if self.branch_fuse:
            self.fused_feat_dimensions = len(self.branch_fuse_idx) * 512 * self.base.block.expansion
            self.softmax_fused = nn.Linear(self.fused_feat_dimensions, self.fuse_dimensions, bias=False)
            self.softmax_fused.apply(self.weights_init_softmax)
        


    def _internal_name_counter(self):
        out="out"+str(self._internal_name_count)
        self._internal_name_count+=1
        return out

    def base_forward(self,x):
        features = self.base(x) # features are a list

        for idx,bname in enumerate(self.branch_name_order):
            features[idx] = self.branches[bname]["gap"](features[idx])
            features[idx] = features[idx].view(features[idx].shape[0], -1)
            features[idx] = self.branches[bname]["emb_linear"](features[idx])

        return features


    def forward_impl(self,x, **kwargs):
        features = self.base_forward(x)
        fused_features = []
        if self.branch_fuse:
            fused_features+= [self.fused_feat_norm(torch.cat([features[idx] for idx in self.branch_fuse_idx], dim=0))]


        outputs = [None]*self.number_outputs
        out_idx= 0
        for b_idx,bname in self.branch_name_order:
            features[b_idx] = self.branches[bname]["feat_norm"](features[b_idx])
            for o_idx,_ in enumerate(self.branches_meta[bname]["output_nameorder"]):
                outputs[out_idx] = self.branches[bname]["softmax"][o_idx](features[b_idx])
                out_idx+=1
        
        fused_outs = []
        if self.branch_fuse:
            fused_outs+= [self.softmax_fused(fused_features[0])]

        return features+fused_features, outputs+fused_outs, []