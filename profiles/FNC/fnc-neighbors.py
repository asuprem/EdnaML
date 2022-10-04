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
    unfilter_file = kwargs.get("unfiltered_output")
    basename = kwargs.get("basename")
    self.neighbor_name = "-".join([basename, neighbor_file])+".txt"
    self.final_neighbor_name = "-".join([basename, neighbor_file])+".json"
    self.unfilter_name = "-".join([basename, unfilter_file])+".json"

    self.neighbor_obj = open(self.neighbor_name, "w")

  def output_step(self, logits, features, secondary):
    # outputs is a list, where the post is in outputs[2], I think????
    # In any case, we need to check ouputs[2] to get the threshold...
    write_list = (secondary[2]["FastKMP-l2"]["distance"] <= secondary[2]["FastKMP-l2"]["threshold"]).long().tolist()
    self.neighbor_obj.write("\n".join([str(item) for item in write_list]) + "\n")

  def end_of_deployment(self):
    self.neighbor_obj.close()

    nobj = open(self.neighbor_name, "r")
    outputobj = open(self.final_neighbor_name, "w")
    uobj = open(self.unfilter_name, "r")

    fls = [nobj, uobj]
    from itertools import zip_longest
    for lines in zip_longest(*fls, fillvalue=""):
      if int(lines[0].strip()) == 1:
        outputobj.write(lines[1])

    outputobj.close()