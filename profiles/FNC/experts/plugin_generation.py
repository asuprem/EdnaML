from ednaml.deploy import BaseDeploy
import ednaml.core.decorators as edna


@edna.register_deployment
class FNCPluginDeployment(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    prediction_scores, pooled_out, outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    
    return prediction_scores, pooled_out, outputs
  def output_setup(self, **kwargs):
    pass
  def output_step(self, logits, features, secondary):
    pass