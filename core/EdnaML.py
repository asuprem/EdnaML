import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary

from utils.SaveMetadata import SaveMetadata



class EdnaML:



    def __init__(self, config:str="config.yaml", mode:str="train", weights:str=None, logger:logging.Logger=None):
        
        
        self.buildConfig(config)
        self.buildSaveMetadata()
        self.makeSaveDirectories()
        self.buildLogger(logger)
        self.printConfiguration()



    def buildConfig(self, config, handler="yaml"):
        self.cfg = kaptan.Kaptan(handler=handler)
        self.cfg = self.cfg.import_config(config)


    def buildSaveMetadata(self):
        self.saveMetadata = SaveMetadata(self.cfg)

    def makeSaveDirectories(self):
        # NOTE: set up the base config file...
        os.makedirs(self.saveMetadata.MODEL_SAVE_FOLDER, exist_ok=True)

    def buildLogger(self, logger):
        self.logger = logger
        # TODO fix the logger...
        # utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    def log(self, message:str, level):
        pass

    def printConfiguration(self):
        self.logger.info("*"*40);self.logger.info("");self.logger.info("")
        self.logger.info("Using the following configuration:")
        self.logger.info(self.cfg.export("yaml", indent=4))
        self.logger.info("");self.logger.info("");self.logger.info("*"*40)