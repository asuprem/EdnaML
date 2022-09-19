from ednaml.models.ModelAbstract import ModelAbstract
from typing import Any, List, Tuple
import torch
from ednaml.utils import layers, locate_class
from torch import TensorType, nn
from transformers import AutoModel

class HFAutoModel(ModelAbstract):
    """For Sequence Classification only.

    Future plans: HF has a pretty structured method for the heads
    So, we can extend the HFAutoModel to use multiple heads...

    Essentially, construct the necessary heads from the base model type 
    (i.e. <ModelName>ForSequenceClassification, i.e. MPNetForSequenceClassification)

    Then, we construct the core model, i.e <ModelName>Model.from_pretrained(), i.e. MPNetModel.from_pretrained()
    Then, we can stick the heads on top.

    Args:
        ModelAbstract (_type_): _description_

    Model_Kwargs:
        NOTE: Since we are loading from pretrained, either MODEL_BASE or MODEL_KWARGS.from_pretrained MUST be passed
        NOTE: Do NOT use `pretrained_model_name_or_path` as a MODEL_KWARGS field, because this will conflict with passed arguments to AutoModel. 
        See HuggingFace API for details on model_kwargs

    """


    def model_attributes_setup(self, **kwargs):
        self.from_pretrained = kwargs.get("from_pretrained", self.model_base)


    def model_setup(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(self.from_pretrained, **kwargs)

    def foward_impl(self, x, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        response = self.classifier(x,attention_mask = attention_mask, token_type_ids = token_type_ids, position_ids=position_ids, head_mask=head_mask )

        return response.logits, None, [response.attentions]
        