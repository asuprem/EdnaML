import ednaml, torch, os, csv
from ednaml.crawlers import Crawler
from ednaml.deploy.BaseDeploy import BaseDeploy
import ednaml.core.decorators as edna


# For each batch, get the predicted labels, and save the labels in an append text file

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
    file_path = kwargs.get("file_name", "deploy-out.txt")
    self.fobj = open(file_path, "w")

  def output_step(self, logits, features, secondary):
    predicted_label = logits.tolist()
    l2_predicted_label = secondary["KMP-l2"]["cluster_mean"].tolist()
    l2_dist = secondary["KMP-l2"]["distance"].to(torch.float32).tolist()
    cos_predicted_label = secondary["KMP-cos"]["cluster_mean"].tolist()
    cos_dist = secondary["KMP-cos"]["distance"].to(torch.float32).tolist()
    l_score = secondary["RL-midas"]["l_score"]
    smooth_l_score = secondary["RL-midas"]["smooth_l_score"]
    l_threshold = secondary["RL-midas"]["l_threshold"]
    smooth_l_threshold = secondary["RL-midas"]["smooth_l_threshold"]

    output_list = [",".join(map(str,item)) for item in zip(
      predicted_label,
      l2_predicted_label,
      l2_dist,
      cos_predicted_label,
      cos_dist,
      l_score,
      smooth_l_score,
      l_threshold,
      smooth_l_threshold
      )]
    self.fobj.write("\n".join(output_list)+"\n")

#@edna.register_deployment
class NELADeployPluginSetup(NELADeploy):
  def output_setup(self, **kwargs):
    pass
  def output_step(self, logits, features, secondary):
    pass



import click
@click.argument("config")
@click.argument("mode")
def main(config, deploy, mode):
    from ednaml.core import EdnaDeploy
    ed = EdnaDeploy(config=config, deploy=deploy)
    ed.add("./nela.py")
    ed.addModelClass(main.NELAModel)
    ed.addDeploymentClass(NELADeploy)

    ed.apply(input_size=(ed.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE,ed.cfg.EXECUTION.DATAREADER.DATASET_ARGS["maxlen"]),
          dtypes=[torch.long])

    ed.deploy()

if __name__ == "__main__":
    main()