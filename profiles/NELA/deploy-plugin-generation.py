import ednaml, torch, os, csv
from ednaml.deploy.BaseDeploy import BaseDeploy
import ednaml.core.decorators as edna

@edna.register_deployment
class NELADeploy(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    logits, features, secondary = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    logit = logits.detach().cpu()
    logit_labels = torch.argmax(logit, dim=1)
    #import pdb
    #pdb.set_trace()
    #print(str(logit_labels[0]), "\t", ",".join([str(item) for item in all_input_ids[0].cpu()[:10].tolist()]))
    if len(secondary)>1:
      plugin_post = secondary[2]
    else:
      plugin_post = None
    return logit_labels, logit, plugin_post

  def output_setup(self, **kwargs):
    pass
  def output_step(self, logits, features, secondary):
    pass