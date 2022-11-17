import json, logging, os, shutil
from typing import Dict, List, Tuple
import tqdm
import torch
from torch.utils.data import DataLoader
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.core import EdnaMLContextInformation
from ednaml.crawlers import Crawler
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils.LabelMetadata import LabelMetadata


class BaseDeploy:
    """Base deployment class
    """
    model: ModelAbstract
    data_loader: DataLoader

    global_batch: int
    metadata: Dict[str, str]
    labelMetadata: LabelMetadata
    logger: logging.Logger

    edna_context: EdnaMLContextInformation
    save_frequency: int
    step_save_frequency: int

    def __init__(
        self,
        model: ModelAbstract,
        data_loader: DataLoader,
        epochs: int,
        logger: logging.Logger,
        crawler: Crawler,
        config: EdnaMLConfig,
        labels: LabelMetadata,
        context: EdnaMLContextInformation,
        **kwargs
    ):

        self.model = model
        self.parameter_groups = list(self.model.parameter_groups.keys())
       
        self.data_loader = data_loader
        self.logger = logger
        self.epochs = epochs
        self.global_epoch = 0
        self.global_batch = 0  # Current batch number in the epoch

        self.metadata = {}
        self.labelMetadata = labels
        self.crawler = crawler
        self.config = config

        self.buildMetadata(
            # TODO This is not gonna work with the torchvision wrapper -- ned to fix that; because crawler is not set up for that pattern...?
            crawler=crawler.classes,
            config=json.loads(config.export("json")),
        )

        self.model_is_built = False
        self.edna_context = context

    def buildMetadata(self, **kwargs):
        for keys in kwargs:
            self.metadata[keys] = kwargs.get(keys)

    def apply(
        self,
        step_verbose: int = 5,
        save_directory: str = "./checkpoint/",
        save_backup: bool = False,
        save_frequency: int = 1,
        step_save_frequency: int = 0,
        backup_directory: str = None,
        gpus: int = 1,
        fp16: bool = False,
        model_save_name: str = None,
        logger_file: str = None,
    ):
        self.step_verbose = step_verbose
        self.save_directory = save_directory
        self.save_frequency = save_frequency
        self.step_save_frequency = step_save_frequency
        self.backup_directory = None
        self.model_save_name = model_save_name
        self.logger_file = logger_file
        self.save_backup = save_backup
        if self.save_backup or self.config.SAVE.LOG_BACKUP:
            self.backup_directory = backup_directory
            os.makedirs(self.backup_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)
        self.saveMetadata()

        self.gpus = gpus

        if self.gpus != 1:
            self.logger.warning("Multi-gpu or non-gpu not yet fully supported.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.gpus:
            #self.model.cuda() # moves the model into GPU
            self.logger.info("%i GPUs available"%self.gpus)
        self.model.to(self.device)

        self.fp16 = fp16
        # if self.fp16 and self.apex is not None:
        #    self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.output_setup(**self.config.DEPLOYMENT.OUTPUT_ARGS)   # TODO 
    def saveMetadata(self):
        print("NOT saving metadata. saveMetadata() function not set up.")

    def deploy(self, continue_epoch=0, continue_step = 0, inference = False, ignore_plugins: List[str] = [], execute: bool = True, model_build: bool = None):
        if model_build or (model_build is None and not self.model_is_built):
            self.logger.info("Starting deployment")
            self.logger.info("Logging to:\t%s" % self.logger_file)
            if self.config.SAVE.LOG_BACKUP:
                self.logger.info(
                    "Logs will be backed up to drive directory:\t%s"
                    % self.backup_directory
                )
            
            #self.logger.info("Loading model from saved epoch %i" % (continue_epoch - 1))
            if continue_epoch or continue_step:
                load_epoch = (continue_epoch - 1) if continue_epoch > 0 else 0
                if self.edna_context.MODEL_HAS_LOADED_WEIGHTS:
                    self.logger.info("Weights have already been loaded into model. Skipping loading of epoch-specific weights from Epoch %i Step %i"%(load_epoch, continue_step))
                else:
                    self.load(load_epoch, continue_step, ignore_plugins=ignore_plugins)

            if inference:
                self.model.inference()
            else:
                self.model.eval()
            self.model_is_built = True
        else:
            if model_build is not None and not model_build:
                self.logger.info("Skipping model building due to `model_build=False`")
            elif self.model_is_built:
                self.logger.info("Skipping model building because model is already built. To force, set the `model_build` flag to True in `ed.deploy`")
            else:
                self.logger.info("Skipping model building")

        if execute:
            self.logger.info("Setting up plugin hooks. Plugins will fire during:  %s"%self.config.DEPLOYMENT.PLUGIN.HOOKS)
            self.model.set_plugin_hooks(self.config.DEPLOYMENT.PLUGIN.HOOKS)

            self.logger.info("Executing deployment for  %i epochs" % self.epochs)
            for epoch in range(self.epochs):
                self.logger.info("Starting epoch %i"%self.global_epoch)
                self.model.pre_epoch_hook(epoch=epoch)
                self.data_step()
                self.model.post_epoch_hook(epoch=epoch)
                self.global_epoch = epoch + 1
                if self.global_epoch % self.save_frequency == 0:
                    self.save()
                self.logger.info("Executing end of epoch steps")
                self.end_of_epoch(epoch=epoch)
            self.end_of_deployment()
            self.logger.info("Completed deployment task.")
        else:
            self.logger.info("Skipping execution of deployment task due to `execute = False`.")

    def data_step(self):   
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.data_loader, total=len(self.data_loader), leave=False
            ):    
                batch = self.move_to_device(batch)
                feature_logits, features, secondary_outputs = self.deploy_step(batch)

                self.output_step(feature_logits, features, secondary_outputs)
                # Log Metrics here and inside the model TODO
                self.global_batch += 1
            # No saving state in the middle of a deployment...

    def move_to_device(self, batch) -> Tuple[torch.Tensor]:
        return (item.to(self.device) for item in batch)
        
    def deploy_step(self, batch):   # USER IMPLEMENTS
        batch = tuple(item.cuda() for item in batch)
        data, labels = batch    # TODO move plugins here to allow labels as well!!!!!!!!
        feature_logits, features, secondary_outputs = self.model(data)

        return feature_logits, features, secondary_outputs

    def end_of_epoch(self, epoch: int):
        pass
    
    def end_of_deployment(self):
        pass

    def output_setup(self, **kwargs): # USER IMPLEMENTS; kwargs from config.DEPLOYMENT.OUTPUT_ARGS
        self.logger.info("Warning: No output setup is performed")

    def output_step(self, logits, features, secondary): # USER IMPLEMENTS, ALSO, NEED SOME STEP LOGGING...????????
        if self.global_batch % self.config.LOGGING.STEP_VERBOSE == 0:
            self.logger.info("Warning: No output is generated at step %i"%self.global_batch)

    def load(self, load_epoch, load_step = 0, ignore_plugins: List[str] = []):
        self.logger.info(
            "Loading a model from saved epoch %i, step %i"
            % (load_epoch, load_step)
        )
        model_load = "".join([self.model_save_name, "_epoch%i" % load_epoch, "_step%i" % load_step, ".pth"])
            
        if self.save_backup:
            self.logger.info(
                "Loading model from drive backup."
            )
            model_load_path = os.path.join(self.backup_directory, model_load)
        else:
            self.logger.info(
                "Loading model from local backup."
            )
            model_load_path = os.path.join(self.save_directory, model_load)

        if not (os.path.exists(model_load_path)):
            self.logger.info("Could not find model or training path at %s. Defaulting to not using step parameter."%model_load)
            
            model_load = "".join([self.model_save_name, "_epoch%i" % load_epoch, ".pth"])

            if self.save_backup:
                self.logger.info(
                    "Loading model from drive backup."
                )
                model_load_path = os.path.join(self.backup_directory, model_load)
            else:
                self.logger.info(
                    "Loading model from local backup."
                )
                model_load_path = os.path.join(self.save_directory, model_load)

            if not (os.path.exists(model_load_path)):
                self.logger.info("Final attempt. Could not find model or training path at %s. Not loading."%model_load)
            else:
                self.model.load_state_dict(torch.load(model_load_path))
                self.logger.info(
                    "Finished loading model state_dict from %s" % model_load_path
                )
        else:
            self.model.load_state_dict(torch.load(model_load_path))
            self.logger.info(
                "Finished loading model state_dict from %s" % model_load_path
            )

        
        # Here, we will need to get a list of pickled or serialized plugin objects, then load them into a dictionary, then pass them into model
        # YES!!!!
        plugin_load = self.model_save_name + "_plugins.pth"

        if self.save_backup:
            self.logger.info(
                "Looking for model plugins from drive backup."
            )
            plugin_load_path = os.path.join(self.backup_directory, plugin_load)
        else:
            self.logger.info(
                "Looking for model plugins from local backup."
            )
            plugin_load_path = os.path.join(self.save_directory, plugin_load)

        if os.path.exists(plugin_load_path):
            
            self.model.loadPlugins(plugin_load_path, ignore_plugins=ignore_plugins)
            self.logger.info(
                "Loaded plugins from %s"%plugin_load_path
            )
        else:
            self.logger.info(
                "No plugins found at %s"%plugin_load_path
            )


    def save(self, save_epoch: int = None):
        if save_epoch is None:
            save_epoch = self.global_epoch
        self.logger.info("Performing save at epoch %i"%save_epoch)
        self.logger.info("Saving model plugins.")
        PLUGIN_SAVE_NAME = self.model_save_name + "_plugins.pth"
        # here, we save only the plugins...
        # Do not need model save, or training save, or log save...
        
        # A ModelAbstract can save its own plugins...?
        # No, we get back a plugin serialization object that we use to , well, save the plugin
        # Later, during plugin instantiation, we will need to write the code to load the plugin's save code...
        plugin_save = self.model.savePlugins()

        if len(plugin_save) > 0:
            torch.save(plugin_save, os.path.join(
                self.save_directory, PLUGIN_SAVE_NAME
            ))
            self.logger.info("Saved plugins: %s"%str(plugin_save.keys()))

            if self.save_backup:
                shutil.copy2(
                    os.path.join(self.save_directory, PLUGIN_SAVE_NAME),
                    self.backup_directory,
                )

                self.logger.info(
                    "Performing drive backup of model plugins"
                )
        else:
            self.logger.info("No plugins to save")

       
        