import tqdm
from sklearn.metrics import f1_score
import torch
import numpy as np
from ednaml.models.HFMLMSequenceModel import HFMLMSequenceModel
from ednaml.trainer import BaseTrainer


class HFMLMSequenceTrainer(BaseTrainer):
    model: HFMLMSequenceModel
    def init_setup(self, **kwargs):
        # self.maskedaccuracy = []
        self.softaccuracy = []

    def step(self, batch):
        # batch = tuple(item.cuda() for item in batch)
        (
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_masklm,
            all_annotations,
            all_labels,
        ) = batch
        outputs = self.model(
            all_input_ids,
            token_type_ids=all_token_type_ids,
            attention_mask=all_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            secondary_inputs=all_annotations,  
        )
        # outputs --> class_out, outputs.last_hidden_state, [mlm_out, outputs.attentions]
        logits = outputs[0]
        #features = outputs[1]
        mlm_logits = outputs[2][0]
        #attentions = outputs[2][1]

        logits_loss = self.loss_fn["classification"](
            logits=logits,
            labels=all_labels[:, 0],  # Because there could be multiple labels...
        )
        softmax_accuracy = (logits.max(1)[1] == all_labels[:, 0]).float().mean()

        masked_loss = self.loss_fn["mask_lm"](
            input=mlm_logits.view(
                -1, self.model.model_config.vocab_size
            ),
            target=all_masklm.view(-1),
        )

        

        self.losses["classification"].append(logits_loss.item())
        self.losses["mask_lm"].append(masked_loss.item())
        self.softaccuracy.append(softmax_accuracy.cpu().item())

        return logits_loss+masked_loss

    def evaluate_impl(self):
        logits, labels = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.test_loader, total=len(self.test_loader), leave=False
            ):
                batch = tuple(item.to(self.device) for item in batch)
                (
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_masklm,
                    all_annotations,
                    all_labels,
                ) = batch
                outputs = self.model(
                    all_input_ids,
                    token_type_ids=all_token_type_ids,
                    attention_mask=None,
                    output_attentions=True,
                    output_hidden_states=True,
                    secondary_inputs=all_annotations,
                )
                raw_logits = outputs[0]
                logit = raw_logits.detach().cpu()
                label = all_labels.detach().cpu()
                
                logits.append(logit)
                labels.append(label)

        logits, labels = (torch.cat(logits, dim=0), torch.cat(labels, dim=0))
        self.logger.info("Obtained logits and labels, validation in progress")

        logit_labels = torch.argmax(logits, dim=1)
        accuracy = (logit_labels == labels[:, 0]).sum().float() / float(
            labels[:, 0].size(0)
        )
        micro_fscore = np.mean(f1_score(labels[:, 0], logit_labels, average="micro"))
        weighted_fscore = np.mean(
            f1_score(labels[:, 0], logit_labels, average="weighted")
        )
        self.logger.info("\tAccuracy: {:.3%}".format(accuracy))
        self.logger.info("\tMicro F-score: {:.3f}".format(micro_fscore))
        self.logger.info("\tWeighted F-score: {:.3f}".format(weighted_fscore))

        return logit_labels, labels, logits

    def printStepInformation(self):
        loss_avg = [0.0] * len(self.losses)
        for idx, lossname in enumerate(self.losses):
            loss_avg[idx] += (
                sum(self.losses[lossname][-self.step_verbose :]) / self.step_verbose
            )
        # loss_avg /= self.num_losses
        soft_avg = sum(self.softaccuracy[-100:]) / float(len(self.softaccuracy[-100:]))
        self.logger.info(
            "Epoch{0}.{1}\tMaskedLM: {2:.3f}\Classification: {3:.3f}\tTraining Acc: {4:.3f}".format(
                self.global_epoch,
                self.global_batch,
                loss_avg[0],    # Assume masked_lm loss goes first
                loss_avg[1],
                soft_avg,
            )
        )
    