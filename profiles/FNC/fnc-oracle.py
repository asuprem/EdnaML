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
    extended_file = kwargs.get("extended_output")
    basename = kwargs.get("basename")
    self.oracle_name = "-".join([basename, oracle_file])+".txt"
    self.final_oracle_name = "-".join([basename, oracle_file])+".json"
    self.extended_name = "-".join([basename, extended_file])+".json"
    self.target = kwargs.get("oracle_target", 500)
    self.oracle_obj = open(self.oracle_name, "w")

  def output_step(self, logits, features, secondary):
    # outputs is a list, where the post is in outputs[2], I think????
    # In any case, we need to check ouputs[2] to get the threshold...
    dist = secondary[2]["FastKMP-l2"]["distance"].tolist()
    idx = secondary[2]["FastKMP-l2"]["label"].tolist()
    self.oracle_obj.write("\n".join([",".join([str(item[0]), str(item[1])]) for item in zip(dist, idx)]) + "\n")

  def end_of_deployment(self):
    self.oracle_obj.close()
    

    import pandas as pd
    import math
    import numpy as np
    df = pd.read_csv(self.oracle_name, header=None, names=["distance", "label"])
    df=df.reset_index()

    oracle_per_proxy = math.ceil(self.target / len(df["label"].unique()))
    proxies = df["label"].unique().tolist()

    #for proxy in proxies:
    extras = 0
    indices = []
    for proxy in proxies:
      proxydf = df[df["label"] == proxy]
      proxycount = proxydf.count()[0]
      if proxycount < oracle_per_proxy:
        proxy_select = proxycount
        extras += oracle_per_proxy - proxycount
      else:
        proxy_select = oracle_per_proxy
      proxy_quantiles = np.linspace(0,1,proxy_select, endpoint=False)
      indices += proxydf.quantile(proxy_quantiles, interpolation="nearest")["index"].unique().tolist()
      if len(indices) < proxy_select:
        extras += proxy_select - len(indices)
    sorted_indices = sorted(indices)

    oracle_obj = open(self.final_oracle_name, "w")
    extended_obj = open(self.extended_name, "r")
    current_idx = 0
    current_index = sorted_indices[current_idx]
    for idx, line in enumerate(extended_obj):
      if idx == current_index:
        oracle_obj.write(line)
        current_idx += 1
        if current_idx >= len(sorted_indices):
          break
        current_index = sorted_indices[current_idx]
      if idx%10000 == 0:
        self.logger.debug("Processed %i points"%idx)
    oracle_obj.close()
    extended_obj.close()