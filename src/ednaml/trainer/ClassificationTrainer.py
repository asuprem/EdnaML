import tqdm, json
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
from ednaml.trainer import BaseTrainer
from ednaml.utils import locate_class

class ClassificationTrainer(BaseTrainer):
    def init_setup(self, **kwargs):
        self.softaccuracy = []
        # Initialize all metrics specified in config
        self.metrics = []
        for metric_name, metric_config in kwargs['METRICS'].items(): # Iterate over all metrics
            if metric_config['metric_type'] == 'EdnaML_TorchMetric': # Only consider TorchMetrics for now
                metric_class = locate_class(
                    package='ednaml.metrics',
                    subpackage=metric_config['subpackage'],
                    classfile=metric_config['classfile'],
                    classpackage=metric_config['metric_class'],
                )
                metric = metric_class(
                    metric_name=metric_name,
                    metric_args=metric_config['metric_args'],
                    metric_params=metric_config['metric_params']
                )
                self.metrics.append(metric)
        print(self.metrics)
        print(f'Metrics initialization complete! Created {len(self.metrics)} metric(s).')

    # Steps through a batch of data
    def step(self, batch): 
        batch_kwargs = {}
        (
            img,
            batch_kwargs["labels"],
        ) = batch  
        # logits, features, labels
        batch_kwargs["logits"], batch_kwargs["features"], _ = self.model(img)
        batch_kwargs["epoch"] = self.global_epoch  # For CompactContrastiveLoss

        loss = {loss_name: None for loss_name in self.loss_fn}
        for lossname in self.loss_fn:
            loss[lossname] = self.loss_fn[lossname](**batch_kwargs)
        # if self.fp16 and self.apex is not None:
        #    with self.apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #        scaled_loss.backward()
        # else:
        #    loss.backward()
        lossbackward = sum(loss.values())

        for _, lossname in enumerate(self.loss_fn):
            self.losses[lossname].append(loss[lossname].cpu().item())

        if batch_kwargs["logits"] is not None:
            softmax_accuracy = (
                (batch_kwargs["logits"].max(1)[1] == batch_kwargs["labels"])
                .float()
                .mean()
            )
            self.softaccuracy.append(softmax_accuracy.cpu().item())
        else:
            self.softaccuracy.append(0)

        if self.metrics: # Execute if metrics API usage is specified
            print('Metrics API entry point')
            for metric in self.metrics:
                metric.print_info()
                ## TODO: Feed batch into metric update/save
                metric.update(epoch=self.global_epoch,batch=0,preds=batch_kwargs['logits'].detach(),target=batch_kwargs['labels'].detach())
                metric.print_state()
            print('Metrics API Exit point')
        return lossbackward

    def evaluate_impl(self):
        self.model.eval()
        features, logits, labels = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.test_loader, total=len(self.test_loader), leave=False
            ):
                data, label = batch
                data = data.to(self.device)
                logit, feature, _ = self.model(data)
                feature = feature.detach().cpu()
                logit = logit.detach().cpu()
                features.append(feature)
                logits.append(logit)
                labels.append(label)

        features, logits, labels = (
            torch.cat(features, dim=0),
            torch.cat(logits, dim=0),
            torch.cat(labels, dim=0),
        )
        # Now we compute the loss...
        self.logger.info("Obtained features, validation in progress")
        # for evaluation...

        logit_labels = torch.argmax(logits, dim=1)
        accuracy = (logit_labels == labels).sum().float() / float(
            labels.size(0)
        )
        micro_fscore = np.mean(f1_score(labels, logit_labels, average="micro"))
        weighted_fscore = np.mean(
            f1_score(labels, logit_labels, average="weighted")
        )
        self.logger.info("Accuracy: {:.3%}".format(accuracy))
        self.logger.info("Micro F-score: {:.3f}".format(micro_fscore))
        self.logger.info("Weighted F-score: {:.3f}".format(weighted_fscore))
        return logit_labels, labels, logits

    def saveMetadata(
        self,
    ):
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
