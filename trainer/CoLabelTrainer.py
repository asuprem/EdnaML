import tqdm, json
from collections import defaultdict, OrderedDict
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
import loss.builders
from typing import List

from .BaseTrainer import BaseTrainer

import pdb


class CoLabelTrainer(BaseTrainer):
    def __init__(   self, 
                    model: torch.nn.Module, 
                    loss_fn: List[loss.builders.LossBuilder], 
                    optimizer: torch.optim.Optimizer, loss_optimizer: List[torch.optim.Optimizer], 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, loss_scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    train_loader, test_loader, 
                    epochs: int, skipeval, logger, **kwargs):   #kwargs includes crawler
        
        super(CoLabelTrainer,self).__init__(model, loss_fn, optimizer, loss_optimizer, scheduler, loss_scheduler, train_loader, test_loader, epochs, skipeval, logger)
        
        self.crawler = kwargs.get("crawler", None)
        self.softaccuracy = []
        
    # The train function for the CoLabel model is inherited


    def epoch_step(self, epoch):
        """Trains for an epoch
        """
        for batch in self.train_loader:
            # Set up scheduled LR
            if self.global_batch==0:
                lrs = self.scheduler.get_last_lr(); lrs = sum(lrs)/float(len(lrs))
                self.logger.info("Starting epoch {0} with {1} steps and learning rate {2:2.5E}".format(epoch, len(self.train_loader) - (len(self.train_loader)%10), lrs))
            
            # Step through the batch (including loss.backward())
            self.step(batch)
            self.global_batch += 1

            if (self.global_batch + 1) % self.step_verbose == 0:
                loss_avg=0.0
                for idx in range(self.num_losses):
                    loss_avg += sum(self.loss[idx][-self.step_verbose:]) / self.step_verbose
                loss_avg/=self.num_losses
                soft_avg = sum(self.softaccuracy[-100:]) / float(len(self.softaccuracy[-100:]))
                self.logger.info('Epoch{0}.{1}\tTotal Avg Loss: {2:.3f} Softmax: {3:.3f}'.format(self.global_epoch, self.global_batch, loss_avg, soft_avg))
        
        self.global_batch = 0

        # Step the lr schedule to update the learning rate
        self.scheduler.step()
        for idx in range(self.num_losses):
            if self.loss_scheduler[idx] is not None:
                self.loss_scheduler[idx].step()
        
        self.logger.info('{0} Completed epoch {1} {2}'.format('*'*10, self.global_epoch, '*'*10))
        
        if self.global_epoch % self.test_frequency == 0:
            self.logger.info('Evaluating model at test-frequency')
            self.evaluate()
        if self.global_epoch % self.save_frequency == 0:
            self.logger.info('Saving model at save-frequency')
            self.save()
        self.global_epoch += 1

    # Steps through a batch of data
    def step(self,batch):
        # Switch the model to training mode
        self.model.train()
        self.optimizer.zero_grad()
        for idx in range(self.num_losses):
            if self.loss_optimizer[idx] is not None: # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
                self.loss_optimizer[idx].zero_grad()
        batch_kwargs = {}
        img, batch_kwargs["labels"] = batch # This is the tensor response from collate_fn
        img, batch_kwargs["labels"] = img.cuda(), batch_kwargs["labels"].cuda()
        # logits, features, labels
        batch_kwargs["logits"], batch_kwargs["features"] = self.model(img)
        batch_kwargs["epoch"] = self.global_epoch   # For CompactContrastiveLoss
        
        # TODO fix this with info about how many output are in the model...from the config file!!!!!
        loss=[None]*self.num_losses
        for idx in range(self.num_losses):
            loss[idx] = self.loss_fn[idx](**batch_kwargs)
        #if self.fp16 and self.apex is not None:
        #    with self.apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #        scaled_loss.backward()
        #else:
        #    loss.backward()
        for idx in range(self.num_losses):
            loss[idx].backward()
        self.optimizer.step()
        for idx in range(self.num_losses):
            if self.loss_optimizer[idx] is not None: # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
                self.loss_optimizer[idx].step()
        
        for idx in range(self.num_losses):
            self.loss[idx].append(loss[idx].cpu().item())
        
        if batch_kwargs["logits"] is not None:
            softmax_accuracy = (batch_kwargs["logits"].max(1)[1] == batch_kwargs["labels"]).float().mean()
            self.softaccuracy.append(softmax_accuracy.cpu().item())
        else:
            self.softaccuracy.append(0)


    def evaluate(self):
        self.model.eval()
        features, logits, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                data, label = batch
                data = data.cuda()
                logit, feature  = self.model(data)
                feature = feature.detach().cpu()
                logit = logit.detach().cpu()
                features.append(feature)
                logits.append(logit)
                labels.append(label)

        features, logits, labels = torch.cat(features, dim=0), torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        # Now we compute the loss...
        self.logger.info('Obtained features, validation in progress')
        # for evaluation...
        #pdb.set_trace()

        logit_labels = torch.argmax(logits, dim=1)
        accuracy = (logit_labels==labels).sum().float()/float(labels.size(0))
        micro_fscore = np.mean(f1_score(labels,logit_labels, average='micro'))
        weighted_fscore = np.mean(f1_score(labels,logit_labels, average='weighted'))
        self.logger.info('Accuracy: {:.3%}'.format(accuracy))
        self.logger.info('Micro F-score: {:.3f}'.format(micro_fscore))
        self.logger.info('Weighted F-score: {:.3f}'.format(weighted_fscore))
        return logit_labels, labels, self.crawler.classes

    def saveMetadata(self,):
        self.logger.info("Saving model metadata")
        jMetadata = json.dumps(self.metadata)
        metafile = "metadata.json"
        localmetafile = os.path.join(self.save_directory, metafile)
        if self.save_backup:
            backupmetafile = os.path.join(self.backup_directory, metafile)
        if not os.path.exists(localmetafile):
            with open(localmetafile, "w") as localmetaobj:
                localmetaobj.write(jMetadata)
        self.logger.info("Backing up metadata")
        if self.save_backup:
            shutil.copy2(localmetafile, backupmetafile)
        self.logger.info("Finished metadata backup")