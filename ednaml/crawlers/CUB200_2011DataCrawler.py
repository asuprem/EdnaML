import glob
import math
import os
import random
import ednaml.utils.splits.cub200


class CUB200_2011DataCrawler:
    def __init__(self, data_folder="CUB_200_2011", **kwargs):
        self.metadata = {}

        self.data_folder = data_folder
        self.image_folder = os.path.join(self.data_folder, "images")

        self.logger = kwargs.get("logger")

        self.__verify(self.data_folder)
        self.__verify(self.image_folder)

        self.crawl()

    def __verify(self, folder):
        if not os.path.exists(folder):
            raise IOError(
                "Folder {data_folder} does not exist".format(data_folder=folder)
            )
        else:
            self.logger.info("Found {data_folder}".format(data_folder=folder))

    def crawl(self,):
        self.metadata["train"], self.metadata["test"], self.metadata["query"] = (
            {},
            {},
            {},
        )
        self.__crawl(self.image_folder)

        self.logger.info(
            "Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["train"]["pids"],
                self.metadata["train"]["cids"],
                self.metadata["train"]["imgs"],
            )
        )
        self.logger.info(
            "Test \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["test"]["pids"],
                self.metadata["test"]["cids"],
                self.metadata["test"]["imgs"],
            )
        )
        self.logger.info(
            "Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["query"]["pids"],
                self.metadata["query"]["cids"],
                self.metadata["query"]["imgs"],
            )
        )

    def __crawl(self, image_folder):
        # Data/CUB_200_2011/images contains one folder for each of the 200 classes...
        crawler = []
        class_list = glob.glob(image_folder + "/*")
        for class_folder in class_list:
            folder_name = os.path.basename(class_folder)
            class_name = int(folder_name.split(".")[0])
            image_list = glob.glob(class_folder + "/*.jpg")
            crawler += [(item, class_name, 0) for item in image_list]

        # crawler is a list of 3-tuples. Length of N=number of images.
        # Each tuple is (path/to/image, PID, CID)   PID --> class (from 0-200), CID --> 0
        # CID is a holdover from other crawlers used for re-id task, where cid, or camera-id is required. To maintain compatibility with SequencedGenerator (which expects CID) until I write a generator for Cars196
        # PID is similarity from person-reid, where PID stands for person ID. In this case, it is a unique class
        trainvalsplits = {
            int(item.split(".")[0]): idx
            for idx, item in enumerate(ednaml.utils.splits.cub200.trainval)
        }
        testsplits = {
            int(item.split(".")[0]): idx + len(trainvalsplits)
            for idx, item in enumerate(ednaml.utils.splits.cub200.query)
        }
        train_crawler = [
            (item[0], trainvalsplits[item[1]], item[2])
            for item in crawler
            if item[1] in trainvalsplits
        ]
        test_crawler = [
            (item[0], testsplits[item[1]], item[2])
            for item in crawler
            if item[1] in testsplits
        ]

        random.shuffle(train_crawler)
        split = 0.7
        split_idx = math.ceil(split * len(train_crawler))

        self.metadata["test"]["crawl"] = train_crawler[split_idx:]
        self.metadata["test"]["pids"] = len(trainvalsplits)
        self.metadata["test"]["cids"] = 1
        self.metadata["test"]["imgs"] = len(self.metadata["test"]["crawl"])
        train_crawler = train_crawler[:split_idx]

        self.metadata["query"]["crawl"] = test_crawler
        self.metadata["query"]["pids"] = len(testsplits)
        self.metadata["query"]["cids"] = 1
        self.metadata["query"]["imgs"] = len(self.metadata["query"]["crawl"])

        self.metadata["train"]["crawl"] = train_crawler[:split_idx]
        self.metadata["train"]["pids"] = len(trainvalsplits)
        self.metadata["train"]["cids"] = 1
        self.metadata["train"]["imgs"] = len(train_crawler[:split_idx])
