import tqdm, json
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
import loss.builders
from typing import List
from models.MultiClassificationResnet import MultiClassificationResnet
from crawlers import Crawler
from .BaseTrainer import BaseTrainer
from utils.LabelMetadata import LabelMetadata

import pdb


class MultiClassificationTrainer(BaseTrainer):
    model: MultiClassificationResnet
    def __init__(   self, 
                    model: MultiClassificationResnet, 
                    loss_fn: List[loss.builders.LossBuilder], 
                    optimizer: torch.optim.Optimizer, loss_optimizer: List[torch.optim.Optimizer], 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, loss_scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    train_loader, test_loader, 
                    epochs: int, skipeval, logger, 
                    crawler: Crawler, config, labels: LabelMetadata, **kwargs):
        
        super().__init__(model, loss_fn, optimizer, loss_optimizer, scheduler, loss_scheduler, train_loader, test_loader, epochs, skipeval, logger, crawler, config, labels, **kwargs)
        

        self.softaccuracy = []

        # mapping label names and class names to their index for faster retrieval. 
        # TODO some way to integrate classificationclass in DATAREADER to 
        # labelnames in MODEL, so that there is less redundancy...
        self.model_labelorder = {item:idx for idx,item in enumerate(self.model.output_labels)}
        self.data_labelorder = {item:idx for idx,item in enumerate(self.labelMetadata.labels)}
        
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
                for lossname in self.losses:
                    loss_avg += sum(self.losses[lossname][-self.step_verbose:]) / self.step_verbose
                loss_avg/=self.num_losses
                soft_avg = sum(self.softaccuracy[-100:]) / float(len(self.softaccuracy[-100:]))
                self.logger.info('Epoch{0}.{1}\tTotal Avg Loss: {2:.3f} Softmax: {3:.3f}'.format(self.global_epoch, self.global_batch, loss_avg, soft_avg))
        
        self.global_batch = 0

        # Step the lr schedule to update the learning rate
        self.scheduler.step()
        for lossname in self.loss_fn:
            if self.loss_scheduler[lossname] is not None:
                self.loss_scheduler[lossname].step()
        
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
        for lossname in self.loss_fn:
            if self.loss_optimizer[lossname] is not None:
                self.loss_optimizer[lossname].zero_grad()

        batch_kwargs = {}
        img, batch_kwargs["labels"] = batch # This is the tensor response from collate_fn
        img, batch_kwargs["labels"] = img.cuda(), batch_kwargs["labels"].cuda() # labels are in order of labelnames
        # logits, features, labels
        batch_kwargs["logits"], batch_kwargs["features"], _ = self.model(img)  # logits are in order of output_classnames --> model.output_classnames
        batch_kwargs["epoch"] = self.global_epoch   # For CompactContrastiveLoss
        
        loss={loss_name:None for loss_name in self.loss_fn} 
        for lossname in loss:
            akwargs={}
            akwargs["logits"] = batch_kwargs["logits"][self.model_labelorder[self.loss_fn[lossname].loss_label]] # this looks up the lossname in the outputclass names
            akwargs["labels"] = batch_kwargs["labels"][:, self.data_labelorder[self.loss_fn[lossname].loss_label]] # ^ditto
            akwargs["epoch"] = batch_kwargs["epoch"]
            loss[lossname] = self.loss_fn[lossname](**akwargs)

        lossbackward=sum(loss.values())
        lossbackward.backward()

        

        #for idx in range(self.num_losses):
        #    loss[idx].backward()
        
        self.optimizer.step()
        for lossname in self.loss_fn:
            if self.loss_optimizer[lossname] is not None: # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
                self.loss_optimizer[lossname].step()
        
        for idx,lossname in enumerate(self.loss_fn):
            self.losses[lossname].append(loss[lossname].cpu().item())
        
        #if batch_kwargs["logits"] is not None:
            #softmax_accuracy = (batch_kwargs["logits"].max(1)[1] == batch_kwargs["labels"]).float().mean()
            #self.softaccuracy.append(softmax_accuracy.cpu().item())
        #else:
        self.softaccuracy.append(0) # TODO fix this


    def evaluate_impl(self):
        self.model.eval()        
        features, logits, labels = [], [[] for _ in range(self.model.number_outputs)], []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                data, label = batch
                data = data.cuda()
                logit, feature , _ = self.model(data)
                feature = feature.detach().cpu()
                for idx in range(self.model.number_outputs):
                  logits[idx].append(logit[idx].detach().cpu())
                features.append(feature)
                labels.append(label)
        
        #features, logits, labels = torch.cat(features, dim=0), [torch.cat(logit, dim=0) for logit in logits], torch.cat(labels, dim=0)
        features = torch.cat(features, dim=0)
        logits = [torch.cat(logit, dim=0) for logit in logits]
        labels = torch.cat(labels, dim=0)
        # Now we compute the loss...
        self.logger.info('Obtained features, validation in progress')
        # for evaluation...
        #pdb.set_trace()

        logit_labels = [torch.argmax(logit, dim=1) for logit in logits]
        accuracy = [[] for _ in range(self.model.number_outputs)]
        micro_fscore = [[] for _ in range(self.model.number_outputs)]
        weighted_fscore = [[] for _ in range(self.model.number_outputs)]
        for idx, lossname in enumerate(self.loss_fn):
            accuracy[idx] = (logit_labels[self.model_labelorder[self.loss_fn[lossname].loss_label]]==labels[:,self.data_labelorder[self.loss_fn[lossname].loss_label]]).sum().float()/float(labels.size(0))
            micro_fscore[idx] = np.mean(f1_score(labels[:,self.data_labelorder[self.loss_fn[lossname].loss_label]],logit_labels[self.model_labelorder[self.loss_fn[lossname].loss_label]], average='micro'))
            weighted_fscore[idx] = np.mean(f1_score(labels[:,self.data_labelorder[self.loss_fn[lossname].loss_label]],logit_labels[self.model_labelorder[self.loss_fn[lossname].loss_label]], average='weighted'))
        self.logger.info("Metrics\t"+"\t".join(["%s"%lossname for lossname in self.loss_fn]))
        self.logger.info('Accuracy\t'+'\t'.join(['%s: %0.3f'%(self.labelMetadata.labels[idx], accuracy[idx].item()) for idx in range(self.model.number_outputs)]))
        self.logger.info('M F-Score\t'+'\t'.join(['%s: %0.3f'%(self.labelMetadata.labels[idx], micro_fscore[idx].item()) for idx in range(self.model.number_outputs)]))
        self.logger.info('W F-Score\t'+'\t'.join(['%s: %0.3f'%(self.labelMetadata.labels[idx], weighted_fscore[idx].item()) for idx in range(self.model.number_outputs)]))
        return logit_labels, labels, features

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

