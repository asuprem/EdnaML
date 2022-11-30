from sklearn.metrics import f1_score    
import numpy as np

import torch
from ednaml.deploy.BaseDeploy import BaseDeploy
from ednaml.models.MultiClassificationResnet import MultiClassificationResnet

class MultiClassificationDeploy(BaseDeploy):
    model: MultiClassificationResnet
    def deploy_step(self, batch): 
        data, label = batch
        logit, feature, secondary = self.model(data)
        return logit, feature, (secondary, label)

    def output_setup(self, **kwargs):
        self.model_labelorder = {
            item: idx for idx, item in enumerate(self.model.output_labels)
        }
        self.data_labelorder = {
            item: idx for idx, item in enumerate(self.labelMetadata.labels)
        }


        self.pred_logits, self.pred_labels = (
            [[] for _ in range(self.model.number_outputs)],
            [],
        )
    def output_step(self, logits, features, secondary):
        for idx in range(self.model.number_outputs):
            self.pred_logits[idx].append(logits[idx].detach().cpu())
        self.pred_labels.append(secondary[-1].detach().cpu())

    def end_of_epoch(self, epoch: int):
        logits = [torch.cat(logit, dim=0) for logit in self.pred_logits]
        labels = torch.cat(self.pred_labels, dim=0)


        logit_labels = [torch.argmax(logit, dim=1) for logit in logits]
        accuracy = [[] for _ in range(self.model.number_outputs)]
        micro_fscore = [[] for _ in range(self.model.number_outputs)]
        weighted_fscore = [[] for _ in range(self.model.number_outputs)]
        for idx, labelname in enumerate(self.model_labelorder):
            accuracy[idx] = (
                logit_labels[
                    self.model_labelorder[labelname]
                ]
                == labels[
                    :, self.data_labelorder[labelname]
                ]
            ).sum().float() / float(labels.size(0))
            micro_fscore[idx] = np.mean(
                f1_score(
                    labels[
                        :,
                        self.data_labelorder[labelname],
                    ],
                    logit_labels[
                        self.model_labelorder[labelname]
                    ],
                    average="micro",
                )
            )
            weighted_fscore[idx] = np.mean(
                f1_score(
                    labels[
                        :,
                        self.data_labelorder[labelname],
                    ],
                    logit_labels[
                        self.model_labelorder[labelname]
                    ],
                    average="weighted",
                )
            )
        print(
            "Metrics         \t"
            + "\t".join(["%s" % labelname for labelname in self.model_labelorder])
        )
        print(
            "Accuracy        \t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (self.labelMetadata.labels[idx], accuracy[idx].item())
                    for idx in range(self.model.number_outputs)
                ]
            )
        )
        print(
            "Micro F-Score   \t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (self.labelMetadata.labels[idx], micro_fscore[idx].item())
                    for idx in range(self.model.number_outputs)
                ]
            )
        )
        print(
            "Weighted F-Score\t"
            + "\t".join(
                [
                    "%s: %0.3f"
                    % (
                        self.labelMetadata.labels[idx],
                        weighted_fscore[idx].item(),
                    )
                    for idx in range(self.model.number_outputs)
                ]
            )
        )


    

