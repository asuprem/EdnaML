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
        self.metadata = json.loads(os.path.join(CHECKPOINT_DIRECTORY, "metadata.json"))
        
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
        TRAIN_CLASSES = self.config.get("MODEL.SOFTMAX_DIM", self.config.get("MODEL.SOFTMAX"))
        self.model = model_builder( arch = self.config.get("MODEL.MODEL_ARCH"), \
                                base=self.config.get("MODEL.MODEL_BASE"), \
                                weights=None, \
                                soft_dimensions = TRAIN_CLASSES, \
                                embedding_dimensions = self.config.get("MODEL.EMB_DIM"), \
                                normalization = self.config.get("MODEL.MODEL_NORMALIZATION"), \
                                **model_kwargs_dict)
        self.logger.info("Finished instantiating model with {} architecture".format(self.config.get("MODEL.MODEL_ARCH")))
        
        
        self.model.load_state_dict(torch.load(weight))
        self.model.cuda()
        self.model.eval()
               
        

class CoLabelEnsemble:

    def __init__(self, logger: logging.Logger):
        self.cfg = kaptan.Kaptan(handler='yaml')
        self.ensemble_models:List[CoLabelEnsembleMember] = [] 
        self.logger = logger


    def addModel(self, config, weight):
        self.ensemble_models.append(CoLabelEnsembleMember(config,weight,logger=self.logger))

    def addModels(self, configs, weights):
        for model_idx, (model_config,model_weight) in enumerate(zip(configs, weights)):
            self.addModel(model_config, model_weight)

    def finalizeEnsemble(self,):
        self.ensembleMembers = len(self.ensemble_models)

    def predict(self, dataloader):
        logits= [[]]*self.ensembleMembers
        logit_labels= [None]*self.ensembleMembers
        labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_loader, total=len(self.test_loader), leave=False):
                data, label = batch
                data = data.cuda()
                # For each model NOTE NOTE NOTE
                for idx,colabel_member in enumerate(self.ensemble_models):
                    logit, _  = colabel_member.model(data)
                    logit = logit.detach().cpu()
                    logits[idx].append(logit)
                labels.append(label)
                #feature = feature.detach().cpu()
                #logit = logit.detach().cpu()
                #features.append(feature)
                #logits.append(logit)
                #labels.append(label)

        for idx in range(self.ensembleMembers):
            logits[idx] = torch.cat(logits[idx], dim=0)
            logit_labels[idx] = torch.argmax(logits[idx], dim=1)
        labels=torch.cat(labels, dim=0)
        # Now we compute the loss...
        self.logger.info('Obtained features, validation in progress')
        # for evaluation...
        #pdb.set_trace()

        self.logit_labels = logit_labels
        self.labels = labels
