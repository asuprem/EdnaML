import json, logging, os, shutil
from logging import Logger
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import ednaml.loss.builders
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.crawlers import Crawler
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils.LabelMetadata import LabelMetadata


class BaseTrainer:
    model: ModelAbstract
    loss_fn: Dict[str, ednaml.loss.builders.LossBuilder]  # output-name: lossBuilder
    optimizer: Dict[str, torch.optim.Optimizer]  
    loss_optimizer = Dict[str, List[torch.optim.Optimizer]] 
    scheduler: Dict[str, torch.optim.lr_scheduler._LRScheduler]  
    loss_scheduler: Dict[str, List[torch.optim.lr_scheduler._LRScheduler]] 

    skipeval: bool
    train_loader: DataLoader
    test_loader: DataLoader

    epochs: int
    logger: Logger
    global_batch: int
    global_epoch: int
    num_losses: int
    losses: Dict[str, List[int]] 
    metadata: Dict[str, str]
    labelMetadata: LabelMetadata
    logger: logging.Logger

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

        self.model = model
        self.parameter_groups = list(self.model.parameter_groups.keys())
        self.loss_fn_order = {
            idx: lossbuilder.loss_labelname for idx, lossbuilder in enumerate(loss_fn)
        }
        self.loss_fn = {
            lossbuilder.loss_labelname: lossbuilder for lossbuilder in loss_fn
        }
        self.num_losses = len(self.loss_fn)
        self.losses = {lossname: [] for lossname in self.loss_fn}
        self.loss_optimizer = {
            self.loss_fn_order[idx]: loss_optimizer_content
            for idx, loss_optimizer_content in enumerate(loss_optimizer)
        }
        if type(loss_scheduler) is list:
            self.loss_scheduler = {
                self.loss_fn_order[idx]: loss_scheduler_content
                for idx, loss_scheduler_content in enumerate(loss_scheduler)
            }
        else:
            self.loss_scheduler = {
                self.loss_fn_order[idx]: loss_scheduler
                for idx in range(self.num_losses)
            }

        self.optimizer = {self.parameter_groups[idx]:optimizer_item for idx, optimizer_item in enumerate(optimizer)}
        self.scheduler = {self.parameter_groups[idx]:scheduler_item for idx, scheduler_item in enumerate(scheduler)}
        self.skipeval = skipeval
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.logger = logger

        self.global_batch = 0  # Current batch number in the epoch
        self.global_epoch = 0

        self.metadata = {}
        self.labelMetadata = labels
        self.crawler = crawler
        self.config = config

        self.buildMetadata(
            crawler=crawler.classes, config=json.loads(config.export("json"))
        )

    def buildMetadata(self, **kwargs):
        for keys in kwargs:
            self.metadata[keys] = kwargs.get(keys)

    def setup(
        self,
        step_verbose: int = 5,
        save_frequency: int = 5,
        test_frequency: int = 5,
        save_directory: str = "./checkpoint/",
        save_backup: bool = False,
        backup_directory: str = None,
        gpus: int = 1,
        fp16: bool = False,
        model_save_name: str = None,
        logger_file: str = None,
    ):
        self.step_verbose = step_verbose
        self.save_frequency = save_frequency
        self.test_frequency = test_frequency
        self.save_directory = save_directory
        self.backup_directory = None
        self.model_save_name = model_save_name
        self.logger_file = logger_file
        self.save_backup = save_backup
        if self.save_backup:
            self.backup_directory = backup_directory
            os.makedirs(self.backup_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)
        self.saveMetadata()

        self.gpus = gpus

        if self.gpus != 1:
            raise NotImplementedError()

        self.model.cuda()

        self.fp16 = fp16
        # if self.fp16 and self.apex is not None:
        #    self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')

    def saveMetadata(self):
        print("NOT saving metadata. saveMetadata() function not set up.")

    def save(self):
        self.logger.info("Saving model, optimizer, and scheduler.")
        MODEL_SAVE = self.model_save_name + "_epoch%i" % self.global_epoch + ".pth"
        TRAINING_SAVE = (
            self.model_save_name + "_epoch%i" % self.global_epoch + "_training.pth"
        )

        save_dict = {}
        save_dict["optimizer"] = {
            pgn:self.optimizer[pgn].state_dict() for pgn in self.parameter_groups
        }
        save_dict["scheduler"] = {
            pgn:self.scheduler[pgn].state_dict() for pgn in self.parameter_groups
        }
        save_dict["loss_fn"] = {
            lossname: self.loss_fn[lossname].state_dict() for lossname in self.loss_fn
        }
        save_dict["loss_optimizer"] = {
            lossname: (
                self.loss_optimizer[lossname].state_dict()
                if self.loss_optimizer[lossname] is not None
                else None
            )
            for lossname in self.loss_optimizer
        }
        save_dict["loss_scheduler"] = {
            lossname: (
                self.loss_scheduler[lossname].state_dict()
                if self.loss_scheduler[lossname] is not None
                else None
            )
            for lossname in self.loss_scheduler
        }

        # save_dict["loss_optimizer"] = [self.loss_optimizer[idx].state_dict() if self.loss_optimizer[idx] is not None else None for idx in range(self.num_losses)]
        # save_dict["loss_scheduler"] = [self.loss_scheduler[idx].state_dict() if self.loss_scheduler[idx] is not None else None for idx in range(self.num_losses)]

        torch.save(
            self.model.state_dict(), os.path.join(self.save_directory, MODEL_SAVE)
        )
        torch.save(save_dict, os.path.join(self.save_directory, TRAINING_SAVE))

        if self.save_backup:
            shutil.copy2(
                os.path.join(self.save_directory, MODEL_SAVE), self.backup_directory
            )
            shutil.copy2(
                os.path.join(self.save_directory, TRAINING_SAVE), self.backup_directory
            )

            self.logger.info(
                "Performing drive backup of model, optimizer, and scheduler."
            )

            LOGGER_SAVE = os.path.join(self.backup_directory, self.logger_file)
            if os.path.exists(LOGGER_SAVE):
                os.remove(LOGGER_SAVE)
            shutil.copy2(
                os.path.join(self.save_directory, self.logger_file), LOGGER_SAVE
            )

    def load(self, load_epoch):
        self.logger.info(
            "Resuming training from epoch %i. Loading saved state from %i"
            % (load_epoch + 1, load_epoch)
        )
        model_load = self.model_save_name + "_epoch%i" % load_epoch + ".pth"
        training_load = self.model_save_name + "_epoch%i" % load_epoch + "_training.pth"

        if self.save_backup:
            self.logger.info(
                "Loading model, optimizer, and scheduler from drive backup."
            )
            model_load_path = os.path.join(self.backup_directory, model_load)
            training_load_path = os.path.join(self.backup_directory, training_load)

        else:
            self.logger.info(
                "Loading model, optimizer, and scheduler from local backup."
            )
            model_load_path = os.path.join(self.save_directory, model_load)
            training_load_path = os.path.join(self.save_directory, training_load)

        self.model.load_state_dict(torch.load(model_load_path))
        self.logger.info("Finished loading model state_dict from %s" % model_load_path)

        checkpoint = torch.load(training_load_path)
        for pgn in self.parameter_groups:
            self.optimizer[pgn].load_state_dict(
                checkpoint["optimizer"][pgn]
            )

            self.scheduler[pgn].load_state_dict(
                checkpoint["scheduler"][pgn]
            )
        self.logger.info(
            "Finished loading optimizer state_dict from %s" % training_load_path
        )
        self.logger.info(
            "Finished loading scheduler state_dict from %s" % training_load_path
        )

        for lossname in self.loss_fn:
            self.loss_fn[lossname].load_state_dict(checkpoint["loss_fn"][lossname])
            if self.loss_optimizer[lossname] is not None:
                self.loss_optimizer[lossname].load_state_dict(
                    checkpoint["loss_optimizer"][lossname]
                )
            if self.loss_scheduler[lossname] is not None:
                self.loss_scheduler[lossname].load_state_dict(
                    checkpoint["loss_scheduler"][lossname]
                )

            self.logger.info(
                "Finished loading loss state_dict from %s" % training_load_path
            )

        # for idx in range(self.num_losses):
        #    if self.loss_optimizer[idx] is not None:
        #        self.loss_optimizer[idx].load_state_dict(checkpoint["loss_optimizer"][idx])
        #    if self.loss_scheduler[idx] is not None:
        #        self.loss_scheduler[idx].load_state_dict(checkpoint["loss_scheduler"][idx])

    def train(self, continue_epoch=0):
        self.logger.info("Starting training")
        self.logger.info("Logging to:\t%s" % self.logger_file)
        self.logger.info(
            "Models will be saved to local directory:\t%s" % self.save_directory
        )
        if self.save_backup:
            self.logger.info(
                "Models will be backed up to drive directory:\t%s"
                % self.backup_directory
            )
        self.logger.info(
            "Models will be saved with base name:\t%s_epoch[].pth"
            % self.model_save_name
        )
        self.logger.info(
            "Optimizers will be saved with base name:\t%s_epoch[]_optimizer.pth"
            % self.model_save_name
        )
        self.logger.info(
            "Schedulers will be saved with base name:\t%s_epoch[]_scheduler.pth"
            % self.model_save_name
        )

        if continue_epoch > 0:
            load_epoch = continue_epoch - 1
            self.load(load_epoch)

        if not self.skipeval:
            self.logger.info("Performing initial evaluation...")
            self.initial_evaluate()
        else:
            self.logger.info("Skipping initial evaluation.")

        self.logger.info("Starting training from %i" % continue_epoch)
        for epoch in range(self.epochs):
            if epoch >= continue_epoch:
                self.epoch_step(epoch)
            else:
                self.global_epoch = epoch + 1

    def initial_evaluate(self):
        """Evaluation of model before we start training
        """
        self.evaluate()

    def epoch_step(self, epoch):
        """Trains model for an epoch.
        """
        for batch in self.train_loader:
            if self.global_batch == 0:
                self.printOptimizerLearningRates()
            
            self.model.train()
            self.zeroGradOptimizers()
            self.zeroGradLossOptimizers()

            self.step(batch)
            
            self.global_batch+=1

            if (self.global_batch + 1) % self.step_verbose == 0:
                self.printStepInformation()

        self.global_batch = 0
        self.stepSchedulers()
        self.stepLossSchedulers()

        self.logger.info(
            "{0} Completed epoch {1} {2}".format("*" * 10, self.global_epoch, "*" * 10)
        )

        if self.global_epoch % self.test_frequency == 0:
            self.logger.info("Evaluating model at test-frequency")
            self.evaluate()
        if self.global_epoch % self.save_frequency == 0:
            self.logger.info("Saving model at save-frequency")
            self.save()
        self.global_epoch += 1

    def printStepInformation(self):
        loss_avg = 0.0
        for lossname in self.losses:
            loss_avg += (
                sum(self.losses[lossname][-self.step_verbose :])
                / self.step_verbose
            )
        loss_avg /= self.num_losses
        soft_avg = sum(self.softaccuracy[-100:]) / float(
            len(self.softaccuracy[-100:])
        )
        self.logger.info(
            "Epoch{0}.{1}\tTotal Avg Loss: {2:.3f} Softmax: {3:.3f}".format(
                self.global_epoch, self.global_batch, loss_avg, soft_avg
            )
        )

    def evaluate(self):
        logit_labels, true_labels, features = self.evaluate_impl()
        return logit_labels, true_labels, self.crawler.classes, features

    def evaluate_impl(self):
        raise NotImplementedError


    def zeroGradOptimizers(self):
        for optim in self.optimizer:
            self.optimizer[optim].zero_grad()

    def zeroGradLossOptimizers(self):
        for lossname in self.loss_fn:
            if self.loss_optimizer[lossname] is not None:
                self.loss_optimizer[lossname].zero_grad()

    def stepOptimizers(self):
        for optim in self.optimizer:
            self.optimizer[optim].step()

    def stepLossOptimizers(self):
        for lossname in self.loss_fn:
            if (
                self.loss_optimizer[lossname] is not None
            ):  # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
                self.loss_optimizer[lossname].step()

    def stepSchedulers(self):
        for scheduler in self.scheduler:
            self.scheduler[scheduler].step()

    def stepLossSchedulers(self):
        for lossname in self.loss_fn:
            if self.loss_scheduler[lossname] is not None:
                self.loss_scheduler[lossname].step()

    def printOptimizerLearningRates(self):
        for param_group_name in self.optimizer:
            lrs = self.scheduler[param_group_name].get_last_lr()
            lrs = sum(lrs) / float(len(lrs))
            self.logger.info(
                    "Parameter Group `{0}`: Starting epoch {1} with {2} steps and learning rate {3:2.5E}".format(
                        param_group_name,
                        self.global_epoch,
                        len(self.train_loader) - (len(self.train_loader) % 10),
                        lrs,
                    )
                )




