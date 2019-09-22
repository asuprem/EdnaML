# Designed to work with SequencedGenerator
import glob
import os
import re


class CUB200_2011DataCrawler:
  def __init__(self,data_folder="CUB_200_2011",  **kwargs):
    self.metadata = {}
    # train_folder="cars_train", test_folder="cars_test", query_folder="",

    self.data_folder = data_folder
    self.image_folder = os.path.join(self.data_folder, "images")
    #self.test_folder = os.path.join(self.data_folder, test_folder)
    #self.query_folder = os.path.join(self.data_folder, query_folder)

    self.logger = kwargs.get("logger")

    self.__verify(self.data_folder)
    self.__verify(self.image_folder)
    # self.__verify(self.test_folder)
    # self.__verify(self.query_folder)

    self.crawl()

  def __verify(self,folder):
    if not os.path.exists(folder):
      raise IOError("Folder {data_folder} does not exist".format(data_folder=folder))
    else:
      self.logger.info("Found {data_folder}".format(data_folder = folder))

  def crawl(self,):
    self.metadata["train"], self.metadata["test"], self.metadata["query"] = {}, {}, {}
    self.metadata["train"]["crawl"], self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"] = self.__crawl(self.image_folder)
    # This is here to be compatible with generators.SequencedGenerator
    self.metadata["test"]["crawl"], self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"] = [], 0, 0, 0
    self.metadata["query"]["crawl"], self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"] = [], 0, 0, 0
    #self.__crawl(self.query_folder)

    self.logger.info("Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"]))
    self.logger.info("Test \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"]))
    self.logger.info("Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"]))

  def __crawl(self,image_folder):
    # Data/CUB_200_2011/images contains one folder for each of the 200 classes...
    crawler = []
    class_list = glob.glob(image_folder+"/*")
    for class_folder in class_list:
        folder_name = os.path.basename(class_folder)
        class_name = int(folder_name.split(".")[0])
        image_list = glob.glob(class_folder + "/*.jpg")
        crawler += [(item, class_name, 0) for item in image_list]
        
    # crawler is a list of 3-tuples. Length of N=number of images.
    # Each tuple is (path/to/image, PID, CID)   PID --> class (from 0-200), CID --> 0
    # CID is a holdover from other crawlers used for re-id task, where cid, or camera-id is required. To maintain compatibility with SequencedGenerator (which expects CID) until I write a generator for Cars196 
    # PID is similarity from person-reid, where PID stands for person ID. In this case, it is a unique class
    return crawler, 200, 1, len(crawler)
