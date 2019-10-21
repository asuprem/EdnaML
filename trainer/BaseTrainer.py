import torch
import os
import shutil
import loss.builders

class BaseTrainer:
    def __init__(self):
        raise NotImplementedError()

    def __init__(   self, 
                    model: torch.nn.Module, 
                    loss_fn: loss.builders.LossBuilder, 
                    optimizer: torch.optim.Optimizer, loss_optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, loss_scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    train_loader, test_loader, 
                    epochs, logger):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_optimizer = loss_optimizer
        self.scheduler = scheduler
        self.loss_scheduler = loss_scheduler
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.epochs = epochs
        self.logger = logger

        self.global_batch = 0
        self.global_epoch = 0

        self.loss = []

    def setup(self, step_verbose = 5, save_frequency = 5, test_frequency = 5, \
                save_directory = './checkpoint/', save_backup = False, backup_directory = None, gpus=1,\
                fp16 = False, model_save_name = None, logger_file = None):
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

        self.gpus = gpus

        if self.gpus != 1:
            raise NotImplementedError()
        
        self.model.cuda()
        
        self.fp16 = fp16
        if self.fp16 and self.apex is not None:
            self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')

    def save(self):
        self.logger.info("Saving model, optimizer, and scheduler.")
        MODEL_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '.pth'
        OPTIM_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer.pth'
        SCHEDULER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler.pth'
        LOSS_SAVE = self.model_save_name + "_epoch%i"%self.global_epoch + "_loss.pth"
        LOSS_OPTIMIZER_SAVE = self.model_save_name + "_epoch%i"%self.global_epoch + "_loss_optimizer.pth"
        LOSS_SCHEDULER_SAVE = self.model_save_name + "_epoch%i"%self.global_epoch + "_loss_scheduler.pth"

        torch.save(self.model.state_dict(), os.path.join(self.save_directory, MODEL_SAVE))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_directory, OPTIM_SAVE))
        torch.save(self.scheduler.state_dict(), os.path.join(self.save_directory, SCHEDULER_SAVE))
        torch.save(self.loss_fn.state_dict(), os.path.join(self.save_directory, LOSS_SAVE))

        if self.loss_optimizer is not None: # For loss funtions with empty parameters
            torch.save(self.loss_optimizer.state_dict(), os.path.join(self.save_directory, LOSS_OPTIMIZER_SAVE))
        if self.loss_scheduler is not None: # For loss funtions with empty parameters
            torch.save(self.loss_scheduler.state_dict(), os.path.join(self.save_directory, LOSS_SCHEDULER_SAVE))

        if self.save_backup:
            shutil.copy2(os.path.join(self.save_directory, MODEL_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, OPTIM_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, LOSS_SAVE), self.backup_directory)
            if self.loss_optimizer is not None: # For loss funtions with empty parameters
                shutil.copy2(os.path.join(self.save_directory, LOSS_OPTIMIZER_SAVE), self.backup_directory)
            if self.loss_scheduler is not None: # For loss funtions with empty parameters
                shutil.copy2(os.path.join(self.save_directory, LOSS_SCHEDULER_SAVE), self.backup_directory)
            self.logger.info("Performing drive backup of model, optimizer, and scheduler.")
            
            LOGGER_SAVE = os.path.join(self.backup_directory, self.logger_file)
            if os.path.exists(LOGGER_SAVE):
                os.remove(LOGGER_SAVE)
            shutil.copy2(os.path.join(self.save_directory, self.logger_file), LOGGER_SAVE)
    
    def load(self, load_epoch):
        self.logger.info("Resuming training from epoch %i. Loading saved state from %i"%(load_epoch+1,load_epoch))
        model_load = self.model_save_name + '_epoch%i'%load_epoch + '.pth'
        optim_load = self.model_save_name + '_epoch%i'%load_epoch + '_optimizer.pth'
        scheduler_load = self.model_save_name + '_epoch%i'%load_epoch + '_scheduler.pth'
        loss_load = self.model_save_name + "_epoch%i"%load_epoch + "_loss.pth"
        loss_optimizer_load = self.model_save_name + "_epoch%i"%load_epoch + "_loss_optimizer.pth"
        loss_scheduler_load = self.model_save_name + "_epoch%i"%load_epoch + "_loss_scheduler.pth"

        if self.save_backup:
            self.logger.info("Loading model, optimizer, and scheduler from drive backup.")
            model_load_path = os.path.join(self.backup_directory, model_load)
            optim_load_path = os.path.join(self.backup_directory, optim_load)
            scheduler_load_path = os.path.join(self.backup_directory, scheduler_load)
            loss_load_path = os.path.join(self.backup_directory, loss_load)
            loss_optimizer_load_path = os.path.join(self.backup_directory, loss_optimizer_load)
            loss_scheduler_load_path = os.path.join(self.backup_directory, loss_scheduler_load)
        else:
            self.logger.info("Loading model, optimizer, and scheduler from local backup.")
            model_load_path = os.path.join(self.save_directory, model_load)
            optim_load_path = os.path.join(self.save_directory, optim_load)
            scheduler_load_path = os.path.join(self.save_directory, scheduler_load)
            loss_load_path = os.path.join(self.save_directory, loss_load)
            loss_optimizer_load_path = os.path.join(self.save_directory, loss_optimizer_load)
            loss_scheduler_load_path = os.path.join(self.save_directory, loss_scheduler_load)

        self.model.load_state_dict(torch.load(model_load_path))
        self.logger.info("Finished loading model state_dict from %s"%model_load_path)
        self.optimizer.load_state_dict(torch.load(optim_load_path))
        self.logger.info("Finished loading optimizer state_dict from %s"%optim_load_path)
        self.scheduler.load_state_dict(torch.load(scheduler_load_path))
        self.logger.info("Finished loading scheduler state_dict from %s"%scheduler_load_path)
        self.loss_fn.load_state_dict(torch.load(loss_load_path))
        self.logger.info("Finished loading loss state_dict from %s"%loss_load_path)

        if self.loss_optimizer is not None: # For loss funtions with empty parameters
            self.loss_optimizer.load_state_dict(torch.load(loss_optimizer_load_path))
        if self.loss_scheduler is not None: # For loss funtions with empty parameters
            self.loss_scheduler.load_state_dict(torch.load(loss_scheduler_load_path))
        self.logger.info("Finished loading loss state_dict from %s"%loss_load_path)

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()