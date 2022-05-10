import tqdm, json
from sklearn.metrics import f1_score
import shutil
import os
import torch
import numpy as np
import ednaml.loss.builders
from typing import List
from ednaml.models.MultiBranchResnet import MultiBranchResnet
from ednaml.crawlers import Crawler
from ednaml.trainer import BaseTrainer
from ednaml.utils.LabelMetadata import LabelMetadata


class MultiBranchTrainer(BaseTrainer):
    model: MultiBranchResnet

    def __init__(
        self,
        model: MultiBranchResnet,
        loss_fn: List[ednaml.loss.builders.LossBuilder],
        optimizer: torch.optim.Optimizer,
        loss_optimizer: List[torch.optim.Optimizer],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader,
        test_loader,
        epochs: int,
        skipeval,
        logger,
        crawler: Crawler,
        config,
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

        # mapping label names and class names to their index for faster retrieval.
        # TODO some way to integrate classificationclass in DATAREADER to
        # labelnames in MODEL, so that there is less redundancy...
        self.model_labelorder = {
            item: idx for idx, item in enumerate(self.model.model_labelorder)
        }
        self.model_nameorder = {
            item: idx for idx, item in enumerate(self.model.model_nameorder)
        }
        self.data_labelorder = {
            item: idx for idx, item in enumerate(self.labelMetadata.labels)
        }
        self.model_name_label_map = {
            item[0]: item[1]
            for item in zip(
                self.model.model_nameorder, self.model.model_labelorder
            )
        }

    # Steps through a batch of data
    def step(self, batch):
        batch_kwargs = {}
        (
            img,
            batch_kwargs["labels"],
        ) = batch  # This is the tensor response from collate_fn
        img, batch_kwargs["labels"] = (
            img.cuda(),
            batch_kwargs["labels"].cuda(),
        )  # labels are in order of labelnames
        # logits, features, labels
        batch_kwargs["logits"], batch_kwargs["features"], _ = self.model(
            img
        )  # logits are in order of output_classnames --> model.output_classnames
        batch_kwargs["epoch"] = self.global_epoch  # For CompactContrastiveLoss

        loss = {loss_name: None for loss_name in self.loss_fn}
        for lossname in loss:  # this loss targets a specific output

            akwargs = {}

            akwargs["logits"] = batch_kwargs["logits"][
                self.model_nameorder[lossname]
            ]  # this looks up the lossname in the outputclass names

            # Check if this is a loss for a soft target
            if lossname in self.model.soft_names:
                # we need to adjust the labels...otherwise, we do not need to adjust labels...
                akwargs["labels"] = batch_kwargs["logits"][
                    self.model_nameorder[self.model.soft_target_output_source]
                ]
            else:
                akwargs["labels"] = batch_kwargs["labels"][
                    :, self.data_labelorder[self.model_name_label_map[lossname]]
                ]  # ^ditto
            akwargs["epoch"] = batch_kwargs["epoch"]
            loss[lossname] = self.loss_fn[lossname](**akwargs)

        lossbackward = sum(loss.values())
        lossbackward.backward()

        # for idx in range(self.num_losses):
        #    loss[idx].backward()

        self.stepOptimizers()
        self.stepLossOptimizers()

        for idx, lossname in enumerate(self.loss_fn):
            self.losses[lossname].append(loss[lossname].cpu().item())

        # if batch_kwargs["logits"] is not None:
        # softmax_accuracy = (batch_kwargs["logits"].max(1)[1] == batch_kwargs["labels"]).float().mean()
        # self.softaccuracy.append(softmax_accuracy.cpu().item())
        # else:
        self.softaccuracy.append(0)  # TODO fix this

    def evaluate_impl(self):
        self.model.eval()
        features, logits, labels = (
            [[] for _ in range(self.model.feature_count)],
            [[] for _ in range(self.model.output_count)],
            [],
        )
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.test_loader, total=len(self.test_loader), leave=False
            ):
                data, label = batch
                data = data.cuda()
                logit, feature, _ = self.model(data)
                for idx in range(self.model.feature_count):
                    features[idx].append(feature[idx].detach().cpu())
                for idx in range(self.model.output_count):
                    logits[idx].append(logit[idx].detach().cpu())
                labels.append(label)

        # features, logits, labels = torch.cat(features, dim=0), [torch.cat(logit, dim=0) for logit in logits], torch.cat(labels, dim=0)
        features = [torch.cat(feature, dim=0) for feature in features]
        logits = [torch.cat(logit, dim=0) for logit in logits]
        labels = torch.cat(labels, dim=0)
        # Now we compute the loss...
        self.logger.info("Obtained features, validation in progress")
        # for evaluation...

        logit_labels = [torch.argmax(logit, dim=1) for logit in logits]
        accuracy = [[] for _ in range(self.model.output_count)]
        micro_fscore = [[] for _ in range(self.model.output_count)]
        weighted_fscore = [[] for _ in range(self.model.output_count)]

        """
        akwargs["logits"] = batch_kwargs["logits"][self.model_labelorder[self.model_name_label_map[lossname]]] # this looks up the lossname in the outputclass names
        akwargs["labels"] = batch_kwargs["labels"][:, self.data_labelorder[self.model_name_label_map[lossname]]] # ^ditto
        """
        for idx, lossname in enumerate(self.loss_fn):
            accuracy[idx] = (
                logit_labels[self.model_nameorder[lossname]]
                == labels[
                    :, self.data_labelorder[self.model_name_label_map[lossname]]
                ]
            ).sum().float() / float(labels.size(0))
            micro_fscore[idx] = np.mean(
                f1_score(
                    labels[
                        :,
                        self.data_labelorder[
                            self.model_name_label_map[lossname]
                        ],
                    ],
                    logit_labels[self.model_nameorder[lossname]],
                    average="micro",
                )
            )
            weighted_fscore[idx] = np.mean(
                f1_score(
                    labels[
                        :,
                        self.data_labelorder[
                            self.model_name_label_map[lossname]
                        ],
                    ],
                    logit_labels[self.model_nameorder[lossname]],
                    average="weighted",
                )
            )

        self.logger.info(
            "Metrics\t"
            + "\t".join(["%s" % lossname for lossname in self.loss_fn])
        )
        self.logger.info(
            "Accuracy\t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (self.model.model_nameorder[idx], accuracy[idx].item())
                    for idx in range(self.model.output_count)
                ]
            )
        )
        self.logger.info(
            "M F-Score\t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (
                        self.model.model_nameorder[idx],
                        micro_fscore[idx].item(),
                    )
                    for idx in range(self.model.output_count)
                ]
            )
        )
        self.logger.info(
            "W F-Score\t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (
                        self.model.model_nameorder[idx],
                        weighted_fscore[idx].item(),
                    )
                    for idx in range(self.model.output_count)
                ]
            )
        )

        return logit_labels, labels, features

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
