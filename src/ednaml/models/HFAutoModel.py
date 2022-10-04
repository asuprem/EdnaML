from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils import locate_class
from transformers import AutoConfig

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

    MODEL_KWARGS:
        auto_class: Which of the HuggingFace AutoModels to use. Choose from:
            AutoModel, AutoModelForPreTraining, AutoModelWithMLMHead, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification
        from_pretrained: The base for the model. If not provided, will default to value in MODEL_BASE
        **kwargs: Any other HF kwargs for an AutoModel

    Model_Kwargs:
        NOTE: Since we are loading from pretrained, either MODEL_BASE or MODEL_KWARGS.from_pretrained MUST be passed
        NOTE: Do NOT use `pretrained_model_name_or_path` as a MODEL_KWARGS field, because this will conflict with passed arguments to AutoModel. 
        See HuggingFace API for details on model_kwargs

    """


    def model_attributes_setup(self, **kwargs):
        self.from_pretrained = kwargs.get("from_pretrained", self.model_base)
        self.auto_class = kwargs.get("auto_class", "AutoModel")


    def model_setup(self, **kwargs):
        auto_class = locate_class(package="transformers", subpackage=self.auto_class)
        self.autoconfig = AutoConfig.from_pretrained(self.from_pretrained, **kwargs)
        self.encoder = auto_class.from_pretrained(self.from_pretrained, config=self.autoconfig)

    def forward_impl(self, x, 
                        attention_mask=None,
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        labels=None,
                        output_attentions=None,
                        output_hidden_states=None, **kwargs):
        response = self.encoder(x,attention_mask = attention_mask, token_type_ids = token_type_ids, position_ids=position_ids, head_mask=head_mask, 
                                    inputs_embeds = inputs_embeds, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
                                    
        # https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo
        return response.logits, response.hidden_states[-1][:,0,:], [response.attentions]
        