import os
import re
import glob
from typing import Dict


class CoLabelVehicleColorCrawler:
    """Data crawler for VehicleColor dataset

    
    """

    def __init__(self,data_folder="VehicleColor", train_folder="train", test_folder="test", validation_folder="val", **kwargs):
        self.metadata = {}
        self.classes = {}
        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)
        self.test_folder = os.path.join(self.data_folder, test_folder)
        self.val_folder = os.path.join(self.data_folder, validation_folder)
        
        self.logger = kwargs.get("logger")

        self.__verify(self.data_folder)
        self.__verify(self.train_folder)
        self.__verify(self.test_folder)

        self.crawl()

    def __verify(self,folder):
        if not os.path.exists(folder):
            raise IOError("Folder {data_folder} does not exist".format(data_folder=folder))
        else:
            self.logger.info("Found {data_folder}".format(data_folder = folder))

    def crawl(self,):

        self.classes["color"] = self.__getclasses(self.train_folder)

        self.metadata["train"], self.metadata["test"], self.metadata["val"] = {}, {}, {}
        self.metadata["train"]["classes"], self.metadata["test"]["classes"], self.metadata["val"]["classes"] = {}, {}, {}
        self.metadata["train"]["crawl"], self.metadata["train"]["classes"]["color"], self.metadata["train"]["imgs"] = self.__crawl(self.train_folder)
        self.metadata["test"]["crawl"], self.metadata["test"]["classes"]["color"], self.metadata["test"]["imgs"] = self.__crawl(self.test_folder)
        self.metadata["val"]["crawl"], self.metadata["val"]["classes"]["color"], self.metadata["val"]["imgs"] = self.__crawl(self.val_folder)

        #self.logger.info("Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"]))
        #self.logger.info("Test \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"]))
        #self.logger.info("Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"]))

    def __getclasses(self, folder) -> Dict:
        class_dir_list = glob.glob(folder + "/*")
        # NOTE might not work for windows :/
        return {item.split('/')[-1]:idx for idx,item in enumerate(class_dir_list)}

    def __crawl(self, folder):
        crawler = []
        for class_name in self.classes["color"]:
            class_path = os.path.join(folder, class_name)
            imgs = glob.glob(os.path.join(class_path, "*.jpg"))
            imgs = [(item, self.classes["color"][class_name]) for item in imgs]
            crawler+=imgs
        return crawler, len(self.classes["color"]), len(crawler)


