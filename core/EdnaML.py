import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary



class EdnaML:



    def __init__(self, config:str="config.yaml", mode:str="train", weights:str=None, logger:logging.Logger=None):
        
        self.buildLogger(logger)

        self.buildConfig(config)


    def buildConfig(self, config, handler="yaml"):
        self.cfg = kaptan.Kaptan(handler=handler)
        self.cfg = self.cfg.import_config(config)

    def log(self, message:str, level):
        pass