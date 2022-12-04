from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils import locate_class
from torch import nn

class HFMLMSequenceModel(ModelAbstract):
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

        #self.auto_class = "AutoModel"
        self.auto_configs = kwargs.get("config", {})
        self.num_labels = self.metadata.getLabelDimensions()
        self.pool_method = kwargs.get("pool_method", "pooled")
        self.dropout_param = kwargs.get("dropout", 0.1)
        if "albert" in self.model_base:
            from transformers.models.albert.configuration_albert import AlbertConfig
            self.model_config: AlbertConfig = AlbertConfig.from_pretrained(self.from_pretrained, **self.auto_configs)
        else:
            raise NotImplementedError()
       

    def model_setup(self, **kwargs):
        
        

        if "albert" in self.model_base:
            
            from transformers.models.albert.modeling_albert import AlbertMLMHead, AlbertModel
            


            self.encoder = AlbertModel.from_pretrained(self.from_pretrained,
                **self.model_config.to_dict())
            
            
            self.dropout = nn.Dropout(self.model_config.classifier_dropout_prob)
            self.cls_head = nn.Linear(self.model_config.hidden_size, self.num_labels)
            self.mlm_head = AlbertMLMHead(config=self.model_config)
    # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/albert/modeling_albert.py#L934
    def forward_impl(
        self,
        x,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids = x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # Outputs from AlbertModel:
        #return BaseModelOutputWithPooling(
        #    last_hidden_state=sequence_output,
        #    pooler_output=pooled_output,
        #    hidden_states=encoder_outputs.hidden_states,
        #    attentions=encoder_outputs.attentions,
        #)

        """ Don't need to compute loss here...
        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        """
        pooled_drop = self.dropout(outputs.pooler_output)
        class_out = self.cls_head(pooled_drop)
        mlm_out = self.mlm_head(outputs.last_hidden_state)
        return class_out, outputs.last_hidden_state, [mlm_out, outputs.attentions]
