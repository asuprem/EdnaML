import tqdm
from sklearn.metrics import f1_score
import torch
import numpy as np
from ednaml.trainer import BaseTrainer


class HFTrainer(BaseTrainer):

    def init_setup(self, **kwargs):
        #self.maskedaccuracy = []
        self.softaccuracy = []

    def step(self, batch):
        batch = tuple(item.cuda() for item in batch)
        (
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_masklm,
            all_annotations,
            all_labels
        ) = batch
        outputs = self.model(
            all_input_ids,
            token_type_ids=all_token_type_ids,
            attention_mask=all_attention_mask,
            output_attentions = True,
            secondary_inputs=all_annotations       # NOT for HFTrainer! because it only expects specific inputs!
        )

        logits = outputs[0]
        features = outputs[1]
        secondaries = outputs[2]
        

        logits_loss = self.loss_fn["classification"](
            logits=logits, labels=all_labels[:,0]   # Because there could be multiple labels...
        )
        softmax_accuracy = (
                    (logits.max(1)[1] == all_labels[:,0])
                    .float()
                    .mean()
                )

        self.losses["classification"].append(logits_loss.item())
        self.softaccuracy.append(softmax_accuracy.cpu().item())
        
        return logits_loss
    
    def evaluate_impl(self):
        return super().evaluate_impl()

    def evaluate_impl(self):
        logits, labels = [],[]
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.test_loader, total=len(self.test_loader), leave=False
            ):
                batch = tuple(item.cuda() for item in batch)
                (
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_masklm,
                    all_annotations,
                    all_labels
                ) = batch
                outputs = self.model(
                            all_input_ids,
                            token_type_ids=all_token_type_ids,
                            attention_mask=all_attention_mask,
                            output_attentions = True,
                            secondary_inputs=all_annotations       # NOT for HFTrainer! because it only expects specific inputs!
                        )
                raw_logits = outputs[0]
                raw_features = outputs[1]
                raw_secondaries = outputs[2]
                logit = raw_logits.detach().cpu()
                label = all_labels.detach().cpu()

                logits.append(logit)
                labels.append(label)


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