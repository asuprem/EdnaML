import torch
from ednaml.models.Albert import AlbertPreTrainedModel, AlbertModel, AlbertOnlyMLMHead
from ednaml.models.Albert import AlbertConfig
from torch.nn import CrossEntropyLoss
from ednaml.models.ModelAbstract import ModelAbstract
from torch import nn

from ednaml.utils.albert import AlbertEmbeddingAverage, AlbertPooledOutput, AlbertRawCLSOutput
import ednaml.core.decorators  as edna

@edna.register_model
class MiDASExpert(ModelAbstract):
  def __init__(self, base, weights, metadata, normalization, parameter_groups, **kwargs):
    super().__init__(base=base,
          weights=weights,
          metadata=metadata, 
          normalization=normalization,
          parameter_groups=parameter_groups,
          **kwargs)
    
  def model_attributes_setup(self, **kwargs):
    self.config = AlbertConfig(**kwargs)
    self.configargs = self.config.getVars()
    self.num_labels = kwargs.get("num_classes")
    self.pool_method = kwargs.get("pooling", "pooled")
  
  def model_setup(self, **kwargs):
    
    self.dropout = nn.Dropout(0.1 if self.config.hidden_dropout_prob == 0 else self.config.hidden_dropout_prob)
    self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    if self.pool_method == "pooled":
      self.pooler_layer = AlbertPooledOutput()
    elif self.pool_method == "raw":
      self.pooler_layer = AlbertRawCLSOutput()
    elif self.pool_method == "average":
      self.pooler_layer = AlbertEmbeddingAverage()
    else:
      raise NotImplementedError()

    self.init_weights()
    
    self.encoder, errors = AlbertModel.from_pretrained("pytorch_model.bin", config=self.config, output_loading_info=True)
    print("Errors \n\t", errors)
    
        
  def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """

        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight


        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

  def forward_impl(self, x, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    outputs = self.encoder(x,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
    
    pooled_output = self.pooler_layer(outputs)  # TODO -- have an option to either use the pooled output, the original output, or average the embeddings together, i.e. a layer that is either a lambda layer, or does some averaging...
    pooled_output = self.dropout(pooled_output+0.1)
    logits = self.classifier(pooled_output)
    return logits, pooled_output, outputs[2:] # list of k scores; hidden states, attentions...

  def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

  def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


  def partial_load(self, weights_path):
    super().partial_load(self, weights_path)  # For this, we need to look at the from_pretrained function to accurately load the saved weights from .bin...