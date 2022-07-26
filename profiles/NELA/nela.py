import ednaml, torch, os, csv, os, json, glob, click
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna


@edna.register_crawler
class NELACrawler(Crawler):
  def __init__(self, logger = None, data_folder="Data", sub_folder="nela-covid-2020"):
    """Crawls the Data folder with all datasets already extracted to their individual folders.
    Assumes specific file construction: files are separated into splits 
    (train, test, and val), with label subsets fake and true, with naming convention:

    <datasetname>-<labelsubset>-<split>.csv

    """
    logger.info("Crawling %s"%(data_folder))
    
    # set up class metadata
    self.classes = {}
    self.classes["reliability"] = 2
    
    # set up paths
    self.data_folder = os.path.join(data_folder , sub_folder)
    labelfile = os.path.join(self.data_folder, "labels.csv")
    self.data_folder = os.path.join(self.data_folder, sub_folder) # because there is nested
    ndata = "newsdata"
    tweet = "tweet"
    
    self.newsdata = os.path.join(self.data_folder, ndata)
    self.tweetdata = os.path.join(self.data_folder, tweet)

    # obtain source labels
    sourcelabels = {}
    with open(labelfile, "r") as lfile:
      labelobj = csv.reader(lfile)
      header = next(labelobj)
      for row in labelobj:
        sourcelabels[row[0]] = int(row[1])

    # set up content metadata
    self.metadata = {}
    self.metadata["train"] = {}
    self.metadata["test"] = {}
    self.metadata["val"] = {}
    self.metadata["train"]["crawl"] = []
    self.metadata["test"]["crawl"] = []
    self.metadata["val"]["crawl"] = []
    
    
    # Get the tweet data as list, <id, text, url> TODO
    with open(os.path.join(self.tweetdata, "tweet.json")) as obj:
      self.tweetdata = json.load(obj)

    # for each file in newsdata: obtain the titles, and save in label propagated. We willa dd to metadata later...
    newsdataitems = glob.glob(os.path.join(self.newsdata, "*.json"))
    labelsets = {0:[],1:[],2:[]}
    for newsdatafile in newsdataitems:
      newssource = os.path.splitext(os.path.basename(newsdatafile))[0]
      label = sourcelabels.get(newssource,1)
      njson = None
      with open(newsdatafile) as jsonobj:
        njson = json.load(jsonobj)
      labelsets[label] += [item["title"] for item in njson]

    # shuffle
    import random
    random.seed(75837)
    random.shuffle(labelsets[0])
    random.seed(75837)
    random.shuffle(labelsets[1])
    random.seed(75837)
    random.shuffle(labelsets[2])

    #Splits
    splits = 0.8
    reliable_train = int(len(labelsets[0])*0.8)
    unreliable_train = int(len(labelsets[2])*0.8)
    reliable_val = int(len(labelsets[0])*0.1)
    unreliable_val = int(len(labelsets[2])*0.1)
    
    # WE WILL ADJUST LABEL --> 0 is fake, 1 is true
    self.metadata["train"]["crawl"] += [(item, 1) for item in labelsets[0][:reliable_train]]
    self.metadata["train"]["crawl"] += [(item, 0) for item in labelsets[2][:unreliable_train]]

    self.metadata["val"]["crawl"] += [(item, 1) for item in labelsets[0][reliable_train:reliable_train+reliable_val]]
    self.metadata["val"]["crawl"] += [(item, 0) for item in labelsets[2][unreliable_train:unreliable_train+unreliable_val]]

    self.metadata["test"]["crawl"] += [(item, 1) for item in labelsets[0][reliable_train+reliable_val:]]
    self.metadata["test"]["crawl"] += [(item, 0) for item in labelsets[2][unreliable_train+unreliable_val:]]
      

    self.metadata["train"]["classes"] = self.classes
    self.metadata["test"]["classes"] = self.classes
    self.metadata["val"]["classes"] = self.classes


from ednaml.models.Albert import AlbertPreTrainedModel, AlbertModel, AlbertOnlyMLMHead
from ednaml.models.Albert import AlbertConfig
from torch.nn import CrossEntropyLoss
from ednaml.models.ModelAbstract import ModelAbstract
from torch import nn

from ednaml.utils.albert import AlbertEmbeddingAverage, AlbertPooledOutput, AlbertRawCLSOutput
@edna.register_model
class NELAModel(ModelAbstract):
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

from ednaml.trainer import BaseTrainer
from ednaml.utils.LabelMetadata import LabelMetadata
from torch.utils.data import DataLoader
from logging import Logger
from typing import Dict, List
from ednaml.config.EdnaMLConfig import EdnaMLConfig
import numpy as np
from sklearn.metrics import f1_score
import tqdm
import torch

@edna.register_trainer
class NELATrainer(BaseTrainer):
  def __init__(
        self,
        model: ModelAbstract,
        loss_fn: List[ednaml.loss.builders.LossBuilder],
        optimizer: List[torch.optim.Optimizer],
        loss_optimizer: List[torch.optim.Optimizer],
        scheduler: List[torch.optim.lr_scheduler._LRScheduler],
        loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        skipeval: bool,
        logger: Logger,
        crawler: Crawler,
        config: EdnaMLConfig,
        labels: LabelMetadata,
        **kwargs
    ):
    super().__init__(
            model,
            loss_fn,
            optimizer,
            loss_optimizer,
            scheduler,
            loss_scheduler,
            train_loader,
            test_loader,
            epochs,
            skipeval,
            logger,
            crawler,
            config,
            labels,
            **kwargs
        )
    self.softaccuracy = []

  def step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)

    logits = outputs[0]
    logits_loss = self.loss_fn["classification"](
        logits=logits, labels=all_labels
    )
    softmax_accuracy = (
                (logits.max(1)[1] == all_labels)
                .float()
                .mean()
            )

    self.losses["classification"].append(logits_loss.item())
    self.softaccuracy.append(softmax_accuracy.cpu().item())
    
    return logits_loss

  def evaluate_impl(self):
    logits, labels, dlabels = [],[],[]
    with torch.no_grad():
      for batch in tqdm.tqdm(
          self.test_loader, total=len(self.test_loader), leave=False
      ):
        batch = tuple(item.cuda() for item in batch)
        all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
        outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
        logit = outputs[0].detach().cpu()
        label = all_labels.detach().cpu()

        logits.append(logit)
        labels.append(label)


        #accuracy.append(
        #    (torch.argmax(logits.detach().cpu(), dim=1) == all_labels.detach().cpu()).sum() / all_labels.size(0)
        #)
    logits, labels = (
        torch.cat(logits, dim=0),
        torch.cat(labels, dim=0)    )
    self.logger.info("Obtained logits and labels, validation in progress")


    logit_labels = torch.argmax(logits, dim=1)
    accuracy = (logit_labels == labels).sum().float() / float(labels.size(0))
    micro_fscore = np.mean(f1_score(labels, logit_labels, average="micro"))
    weighted_fscore = np.mean(f1_score(labels, logit_labels, average="weighted"))
    self.logger.info("\tAccuracy: {:.3%}".format(accuracy))
    self.logger.info("\tMicro F-score: {:.3f}".format(micro_fscore))
    self.logger.info("\tWeighted F-score: {:.3f}".format(weighted_fscore))

    return logit_labels, labels, logits


  def printStepInformation(self):
        loss_avg = 0.0
        for lossname in self.losses:
            loss_avg += (
                sum(self.losses[lossname][-self.step_verbose :])
                / self.step_verbose
            )
        #loss_avg /= self.num_losses
        soft_avg = sum(self.softaccuracy[-100:]) / float(
            len(self.softaccuracy[-100:])
        )

        self.logger.info(
            "Epoch{0}.{1}\tClassification Loss: {2:.3f}\t Training Accuracy:  {3:.3f}".format(
                self.global_epoch, self.global_batch, loss_avg, soft_avg
                )
        )

import click
@click.argument("config")
@click.argument("mode")
def main(config, mode):
    from ednaml.core import EdnaML
    eml = EdnaML(config=config, mode=mode)
    eml.addCrawlerClass(NELACrawler)
    eml.addModelClass(NELAModel)
    eml.addTrainerClass(NELATrainer)

    eml.apply(input_size=(eml.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE,eml.cfg.EXECUTION.DATAREADER.DATASET_ARGS["maxlen"]),
          dtypes=[torch.long])

    eml.train()

if __name__ == "__main__":
    main()