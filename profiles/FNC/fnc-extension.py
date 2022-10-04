"""
Albert model with a MLM head

"""

import ednaml
from ednaml.crawlers import Crawler
from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.models.Albert import AlbertPreTrainedModel, AlbertModel, AlbertOnlyMLMHead
from ednaml.models.Albert import AlbertConfig
from torch.nn import CrossEntropyLoss
from ednaml.models.ModelAbstract import ModelAbstract
import torch
from torch import nn
import ednaml.core.decorators as edna

from ednaml.utils.layers import GradientReversalLayer
from ednaml.utils.albert import (
    AlbertEmbeddingAverage,
    AlbertPooledOutput,
    AlbertRawCLSOutput,
)


@edna.register_model
class FNCAlbertModeler(ModelAbstract):
    def __init__(
        self, base, weights, metadata, normalization, parameter_groups, **kwargs
    ):
        super().__init__(
            base=base,
            weights=weights,
            metadata=metadata,
            normalization=normalization,
            parameter_groups=parameter_groups,
            **kwargs
        )

    def model_attributes_setup(self, **kwargs):
        self.config = AlbertConfig(**kwargs)
        self.configargs = self.config.getVars()
        self.pool_method = kwargs.get("pooling", "pooled")

    def model_setup(self, **kwargs):
        self.encoder, errors = AlbertModel.from_pretrained(
            "pytorch_model.bin", config=self.config, output_loading_info=True
        )
        print("Errors \n\t", errors)
        self.decoders = AlbertOnlyMLMHead(self.config)
        self.decoders.apply(self._init_weights)

        self.tie_weights()

        if self.pool_method == "pooled":
            self.pooler_layer = AlbertPooledOutput()
        elif self.pool_method == "raw":
            self.pooler_layer = AlbertRawCLSOutput()
        elif self.pool_method == "average":
            self.pooler_layer = AlbertEmbeddingAverage()
        else:
            raise NotImplementedError()

    def tie_weights(self):
        self._tie_or_clone_weights(
            self.decoders, self.encoder.embeddings.word_embeddings
        )

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """

        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

        if hasattr(first_module, "bias") and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                "constant",
                0,
            )

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

    def forward_impl(
        self,
        x,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.encoder(
            x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        prediction_scores = self.decoders(sequence_output)
        if self.inferencing:
            return prediction_scores, self.pooler_layer(outputs)
        pooled_out = self.pooler_layer(outputs)

        return (
            prediction_scores,
            pooled_out,
            outputs,
        )  # list of k scores; hidden states, attentions...

    def partial_load(self, weights_path):
        super().partial_load(
            self, weights_path
        )  # For this, we need to look at the from_pretrained function to accurately load the saved weights from .bin...


from ednaml.trainer import BaseTrainer
from torch.utils.data import DataLoader
from logging import Logger
from typing import Dict, List
from ednaml.config.EdnaMLConfig import EdnaMLConfig
import tqdm


@edna.register_trainer
class FNCAlbertModelerTrainer(BaseTrainer):
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
        self.maskedaccuracy = []
        self.fullaccuracy = []

    def step(self, batch):
        batch = tuple(item.cuda() for item in batch)
        (
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_masklm,
            all_labels,
        ) = batch
        outputs = self.model(
            all_input_ids,
            token_type_ids=all_token_type_ids,
            attention_mask=all_attention_mask,
        )

        pred = outputs[0]
        discrimination = outputs[1]
        l_d_logits = outputs[2]

        # get the size of predictions, then sum masked_loss
        # We will late deal with doing the masked_lm for specific decoders only
        # That, is, compute cross-entropy, then zero out the errors when the dataset does not match decoder
        masked_loss = None
        masked_pred = None 
        masked_acc = None
        
        masked_loss = self.loss_fn["mask_lm"](
            input=pred.view(
                -1, self.model.config.vocab_size
            ),
            target=all_masklm.view(-1),
        )
        masked_pred = (
            (
                pred
                .view(-1, self.model.config.vocab_size)
                .max(1)[1]
                == all_input_ids.view(-1)
            )
            .float()
            .mean()
            .cpu()
        )
        masked_idx = all_masklm.view(-1) == -1
        masked_acc = (
            (
                pred
                .view(-1, self.model.config.vocab_size)[masked_idx]
                .max(1)[1]
                == all_input_ids.view(-1)[masked_idx]
            )
            .float()
            .mean()
            .cpu()
        )
        # zero out the bad losses here...
        lm_loss = masked_loss
        
    

        self.losses["mask_lm"].append(lm_loss.item())
        masked_pred = masked_pred if not torch.isnan(masked_pred) else 0 
        masked_acc = masked_acc if not torch.isnan(masked_acc) else 0 
        self.fullaccuracy.append(masked_pred)
        self.maskedaccuracy.append(masked_acc)

        return lm_loss

    def evaluate_impl(self):
        # 1 because single domain
        
        prediction_acc, maskedlm_acc, unmaskedlm_acc = [], [], []
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
                    all_labels
                ) = batch
                outputs = self.model(
                    all_input_ids,
                    token_type_ids=all_token_type_ids,
                    attention_mask=all_attention_mask,
                )
                pred = outputs[0]
                

                masked_pred = None
                masked_acc = None
                unmasked_acc = None
                masked_pred = (
                    (
                        pred
                        .view(-1, self.model.config.vocab_size)
                        .max(1)[1]
                        == all_input_ids.view(-1)
                    )
                    .float()
                    .mean()
                    .cpu()
                )
                masked_idx = all_masklm.view(-1) == -1
                unmasked_idx = all_masklm.view(-1) != -1
                masked_acc = (
                    (
                        pred
                        .view(-1, self.model.config.vocab_size)[masked_idx]
                        .max(1)[1]
                        == all_input_ids.view(-1)[masked_idx]
                    )
                    .float()
                    .mean()
                    .cpu()
                )
                unmasked_acc = (
                    (
                        pred
                        .view(-1, self.model.config.vocab_size)[unmasked_idx]
                        .max(1)[1]
                        == all_input_ids.view(-1)[
                            unmasked_idx
                        ]
                    )
                    .float()
                    .mean()
                    .cpu()
                )


                masked_pred =  masked_pred if not torch.isnan(masked_pred) else 0 
                masked_acc =  masked_acc if not torch.isnan(masked_acc) else 0 
                unmasked_acc =  unmasked_acc if not torch.isnan(unmasked_acc) else 0 

                prediction_acc.append(masked_pred)
                maskedlm_acc.append(masked_acc)
                unmaskedlm_acc.append(unmasked_acc)

        p_accs = [torch.mean(torch.Tensor(item)).item() for item in [prediction_acc]]
        m_accs = [torch.mean(torch.Tensor(item)).item() for item in [maskedlm_acc]]
        u_accs = [torch.mean(torch.Tensor(item)).item() for item in [unmaskedlm_acc]]

        p_str = "\t\tReconstruction\t" + "\t".join(
            ["Domain {0}: {1:.3f}".format(idx, item) for idx, item in enumerate(p_accs)]
        )
        m_str = "\tMasked Prediction\t" + "\t".join(
            ["Domain {0}: {1:.3f}".format(idx, item) for idx, item in enumerate(m_accs)]
        )
        u_str = "\tUnmasked Prediction\t" + "\t".join(
            ["Domain {0}: {1:.3f}".format(idx, item) for idx, item in enumerate(u_accs)]
        )

        self.logger.info(p_str)
        self.logger.info(m_str)
        self.logger.info(u_str)

        return None, None, None # maybe return predictions or something ...?

    def printStepInformation(self):
        loss_avg = [0.0] * len(self.losses)
        for idx, lossname in enumerate(self.losses):
            loss_avg[idx] += (
                sum(self.losses[lossname][-self.step_verbose :]) / self.step_verbose
            )
        # loss_avg /= self.num_losses
        soft_avg = sum(self.fullaccuracy[-100:]) / float(len(self.fullaccuracy[-100:]))
        mask_avg = sum(self.maskedaccuracy[-100:]) / float(
            len(self.maskedaccuracy[-100:])
        )
        self.logger.info(
            "Epoch{0}.{1}\tMaskedLM: {2:.3f}\tReconstruct: {3:.3f}\tMasked Acc: {4:.3f}".format(
                self.global_epoch,
                self.global_batch,
                loss_avg[0],
                soft_avg,
                mask_avg,
            )
        )


import ednaml, torch, os, csv
from ednaml.deploy.BaseDeploy import BaseDeploy
import h5py

@edna.register_deployment
class FNCTrainingFeaturesDeploy(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    prediction_scores, pooled_out, outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    
    return None, pooled_out.cpu(), None

  def output_setup(self, **kwargs):
    output_file = kwargs.get("feature_file", "training_features")
    self.output_file = output_file + ".h5"
    self.writer = h5py.File(self.output_file, "w")  #we will delete old training features file
    self.written = False
    self.prev_idx = -1
    
  def output_step(self, logits, features: torch.LongTensor, secondary): 
    if self.written:
        feats = features.numpy()
        self.writer["features"].resize((self.writer["features"].shape[0] + feats.shape[0]), axis=0)
        self.writer["features"][-feats.shape[0]:] = feats        
    else:   # First time writing -- we will need to create the dataset.
        self.writer.create_dataset("features", data=features.numpy(), compression = "gzip", chunks=True, maxshape=(None,features.shape[1]))
        self.written = True

    if self.writer["features"].shape[0]%5000 > self.prev_idx:
        self.logger.debug("Chunked %i lines in deployment output %s"%(self.writer["features"].shape[0], self.output_file))
        self.prev_idx = self.writer["features"].shape[0]%5000

  def end_of_epoch(self, epoch: int):
      self.writer.close()

  def end_of_deployment(self):
      pass



