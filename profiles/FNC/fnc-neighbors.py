import ednaml.core.decorators as edna



from ednaml.deploy import BaseDeploy

@edna.register_deployment
class FNCPluginDeployment(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    prediction_scores, pooled_out, outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    
    return prediction_scores, pooled_out, outputs
  def output_setup(self, **kwargs):
    neighbor_file = kwargs.get("neighbor_output")
    basename = kwargs.get("basename")
    neighbor_name = "-".join([basename, neighbor_file])+".json"

    self.neighbor_obj = open(neighbor_name, "w")

  def output_step(self, logits, features, secondary):
    import pdb
    pdb.set_trace()
    pass
    # outputs is a list, where the post is in outputs[2], I think????
    # In any case, we need to check ouputs[2] to get the threshold...