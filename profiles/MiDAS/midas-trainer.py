import ednaml
from ednaml.models.ModelAbstract import ModelAbstract
import torch
from ednaml.crawlers import Crawler
from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.trainer import BaseTrainer
from torch.utils.data import DataLoader
from logging import Logger
from typing import Dict, List
from ednaml.config.EdnaMLConfig import EdnaMLConfig
import numpy as np
from sklearn.metrics import f1_score
import tqdm

import ednaml.core.decorators as edna

@edna.register_trainer
class MiDASExpertTrainer(BaseTrainer):
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
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels, all_datalabels = batch
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
        all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels, all_datalabels = batch
        outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
        logit = outputs[0].detach().cpu()
        label = all_labels.detach().cpu()
        dlabel = all_datalabels.detach().cpu()

        logits.append(logit)
        labels.append(label)
        dlabels.append(dlabel)


        #accuracy.append(
        #    (torch.argmax(logits.detach().cpu(), dim=1) == all_labels.detach().cpu()).sum() / all_labels.size(0)
        #)
    logits, labels, dlabels = (
        torch.cat(logits, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(dlabels, dim=0)
    )
    self.logger.info("Obtained logits and labels, validation in progress")


    logit_labels = torch.argmax(logits, dim=1)
    accuracy = (logit_labels == labels).sum().float() / float(labels.size(0))
    micro_fscore = np.mean(f1_score(labels, logit_labels, average="micro"))
    weighted_fscore = np.mean(f1_score(labels, logit_labels, average="weighted"))
    self.logger.info("\tAccuracy: {:.3%}".format(accuracy))
    self.logger.info("\tMicro F-score: {:.3f}".format(micro_fscore))
    self.logger.info("\tWeighted F-score: {:.3f}".format(weighted_fscore))

    return logit_labels, (labels, dlabels), logits


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
