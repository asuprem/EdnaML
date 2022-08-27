import ednaml.core.decorators as edna



from ednaml.deploy import BaseDeploy

@edna.register_deployment
class FNCOracleBinning(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    prediction_scores, pooled_out, outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    
    return prediction_scores, pooled_out, outputs
  def output_setup(self, **kwargs):
    oracle_file = kwargs.get("oracle_output")
    extended_file = kwargs.get("extended_output", "extended")
    basename = kwargs.get("basename")
    self.oracle_name = "-".join([basename, oracle_file])+".txt"
    self.final_oracle_name = "-".join([basename, oracle_file])+".json"
    self.extended_name = "-".join([basename, extended_file])+".json"

    self.oracle_obj = open(self.oracle_name, "w")

  def output_step(self, logits, features, secondary):
    # outputs is a list, where the post is in outputs[2], I think????
    # In any case, we need to check ouputs[2] to get the threshold...
    dist = secondary[2]["FastKMP-l2"]["distance"].tolist()
    idx = secondary[2]["FastKMP-l2"]["idx"].tolist()
    self.oracle_obj.write("\n".join([",".join([str(item[0]), str(item[1])]) for item in zip(dist, idx)]) + "\n")

  def end_of_deployment(self):
    self.oracle_obj.close()

    """
    oobj = open(self.oracle_name, "r")
    outputobj = open(self.final_oracle_name, "w")
    eobj = open(self.extended_name, "r")

    fls = [oobj, eobj]
    from itertools import zip_longest
    for lines in zip_longest(*fls, fillvalue=""):
      if int(lines[0].strip()) == 1:
        outputobj.write(lines[1])

    outputobj.close()
    """