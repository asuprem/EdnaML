
from ednaml.deploy.BaseDeploy import BaseDeploy

class HFDeploy(BaseDeploy):
  def deploy_step(self, batch):
    #batch = tuple(item.cuda() for item in batch)
    (
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_masklm,
        all_annotations,
        all_labels
    ) = batch
    outputs = self.model(
        all_input_ids,
        token_type_ids=all_token_type_ids,
        attention_mask=all_attention_mask,
        output_attentions = True,
        output_hidden_states = True,
        secondary_inputs=all_annotations       # NOT for HFTrainer! because it only expects specific inputs!
    )

    logits = outputs[0]
    features = outputs[1]
    secondaries = outputs[2]
    
    return logits, features, secondaries
  def output_setup(self, **kwargs):
    pass
  def output_step(self, logits, features, secondary):
    pass