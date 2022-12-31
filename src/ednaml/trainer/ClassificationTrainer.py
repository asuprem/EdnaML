import tqdm, json
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
from ednaml.trainer import BaseTrainer
# Avinash entrypoint

class ClassificationTrainer(BaseTrainer):
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
        self.model_params["logits"] = batch_kwargs["logits"]
        self.model_params["features"] = batch_kwargs["features"]
        self.model_params["epoch"] = batch_kwargs["epoch"]
        self.model_params["step"] = self.global_batch
        self.model_params["labels"] = batch_kwargs["labels"]
        loss = {loss_name: None for loss_name in self.loss_fn}
        for lossname in self.loss_fn:
            loss[lossname] = self.loss_fn[lossname](**batch_kwargs)
            self.model_params["loss"+"lossname"] = loss[lossname].cpu().item()
        lossbackward = sum(loss.values())

        for _, lossname in enumerate(self.loss_fn):
            self.losses[lossname].append(loss[lossname].cpu().item())
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
        accuracy = (logit_labels == labels).sum().float() / float(labels.size(0))
        micro_fscore = np.mean(f1_score(labels, logit_labels, average="micro"))
        weighted_fscore = np.mean(f1_score(labels, logit_labels, average="weighted"))
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
