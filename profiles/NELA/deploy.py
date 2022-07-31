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
    logits, features, (secondary, plugin_pre, plugin_post) = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    logit = logits.detach().cpu()
    logit_labels = torch.argmax(logit, dim=1)
    import pdb
    pdb.set_trace()
    #print(str(logit_labels[0]), "\t", ",".join([str(item) for item in all_input_ids[0].cpu()[:10].tolist()]))
    return logit_labels, logit, None

  def output_setup(self, **kwargs):
    file_path = kwargs.get("file_name", "deploy-out.txt")
    self.fobj = open(file_path, "w")

  def output_step(self, logits, features, secondary):
    self.fobj.write("\n".join([str(item) for item in logits.tolist()])+"\n")





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