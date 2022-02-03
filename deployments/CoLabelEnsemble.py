import os
import logging
from typing import List
import kaptan
import json
import torch
import tqdm
import utils

class CoLabelEnsembleMember:
    def __init__(self,config_path,weight, cfg_handler: kaptan.Kaptan, logger: logging.Logger):
        self.logger=logger
        self.config = self.addConfig(config_path, cfg_handler)

        MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(self.config)
        self.metadata = {}
        if os.path.exists(os.path.join(CHECKPOINT_DIRECTORY, "metadata.json")):
          with open(os.path.join(CHECKPOINT_DIRECTORY, "metadata.json"), 'r') as metafile:
            self.metadata = json.load(metafile)
        
        self.model: torch.nn.Module = None
        self.build_model(weight)
        
        

    def addConfig(self, config_path, cfg_handler: kaptan.Kaptan):
        return cfg_handler.import_config(config_path)


    def build_model(self, weight):
        model_builder = __import__("models", fromlist=["*"])
        model_builder = getattr(model_builder, self.config.get("EXECUTION.MODEL_BUILDER", "colabel_model_builder"))
        self.logger.info("Loaded {} from {} to build CoLabel model".format(self.config.get("EXECUTION.MODEL_BUILDER", "colabel_model_builder"), "models"))

        if type(self.config.get("MODEL.MODEL_KWARGS")) is dict:  # Compatibility with old configs. TODO fix all old configs.
            model_kwargs_dict = self.config.get("MODEL.MODEL_KWARGS")
        else:
            model_kwargs_dict = json.loads(self.config.get("MODEL.MODEL_KWARGS"))


        # NOTE: POTENTIAL BUG HERE -- if softmax/softmax_dim is not specified in config...this needs to be fixed...
        TRAIN_CLASSES = self.config.get("MODEL.SOFTMAX_DIM")
        self.model = model_builder( arch = self.config.get("MODEL.MODEL_ARCH"), \
                                base=self.config.get("MODEL.MODEL_BASE"), \
                                weights=None, \
                                soft_dimensions = TRAIN_CLASSES, \
                                embedding_dimensions = self.config.get("MODEL.EMB_DIM", None), \
                                normalization = self.config.get("MODEL.MODEL_NORMALIZATION"), \
                                **model_kwargs_dict)
        self.logger.info("Finished instantiating model with {} architecture".format(self.config.get("MODEL.MODEL_ARCH")))
        
        
        self.model.load_state_dict(torch.load(weight))
        self.model.cuda()
        self.model.eval()
               
        

class CoLabelEnsemble:

    def __init__(self, stacks, logger: logging.Logger):
        self.cfg = kaptan.Kaptan(handler='yaml')
        self.ensemble_models:List[CoLabelEnsembleMember] = [] 
        self.logger = logger
        self.stacks = stacks+1


    def addModel(self, config, weight):
        self.ensemble_models.append(CoLabelEnsembleMember(config,weight,logger=self.logger))

    def addModels(self, configs, weights):
        for model_idx, (model_config,model_weight) in enumerate(zip(configs, weights)):
            self.addModel(model_config, model_weight)

    def finalizeEnsemble(self,):
        self.ensembleMembers = len(self.ensemble_models)

    def predict(self, dataloader):
        """TODO add a line in config about the number of per-model stacks, e.g. jpeg ensemble. This should correspond to the length of the JPEG thingamajig...
        Then, here, we have the size of ensemble and size of stacks per ensemble member to begin with
        create the overall datastructure --> [ [stack1data, stack2data], [stack1data, stack2data], [stack1data, stack2data]]
        labels [labels]

        Then in argmax, perform the argmax for each stack of data
        Then return the overall structure outside. 

        Outside the main function is where we take the argmax values, and check if they match, per stack. wherever they don't, put a (-1) in the final one.
        Then do the voting majority...

        """
        logits= [[[]]*self.stacks]*self.ensembleMembers
        logit_labels= logits= [[None]*self.stacks]*self.ensembleMembers
        labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                # NOTE data is a tuple, potentially. Or something... YES  a tuple!!!!
                #data, label = batch
                data, label = batch
                
                # This is the multi-jpegs stuff...so inside the model loop, for each model, we evaluate for each jpeg compression level (will be slow, ish, maybe...?)
                numstacks = len(data[0])
                assert(numstacks == self.stacks)
                for stack in range(self.stacks):
                    stackdata= data[0][stack].cuda()
                    # For each model NOTE NOTE NOTE
                    for ensemble_idx,colabel_member in enumerate(self.ensemble_models):
                        logit, _  = colabel_member.model(stackdata) # This is the result of the raw, ensemble, etc
                        logit = logit.detach().cpu()
                        logits[ensemble_idx][stack].append(logit)
                
                labels.append(label)
                #feature = feature.detach().cpu()
                #logit = logit.detach().cpu()
                #features.append(feature)
                #logits.append(logit)
                #labels.append(label)

        for ensemble_idx in range(self.ensembleMembers):
            for stack_idx in range(self.stacks):
                logits[ensemble_idx][stack_idx] = torch.cat(logits[ensemble_idx][stack_idx], dim=0)
                logit_labels[ensemble_idx][stack_idx] = torch.argmax(logits[ensemble_idx][stack_idx], dim=1)
        labels=torch.cat(labels, dim=0)
        # Now we compute the loss...
        self.logger.info('Obtained features, validation in progress')
        # for evaluation...
        #pdb.set_trace()

        self.logit_labels = logit_labels
        self.labels = labels
