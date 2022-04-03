import glob
import math
import os
import random

import ednaml.utils.splits.sun


class SUNDataCrawler:
    def __init__(self, data_folder="SUNAttributeDB_Images", **kwargs):
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
        # Data/CUB_200_2011/images contains folders alphabetically...
        traincrawler = []
        querycrawler = []

        for idx, class_name in enumerate(ednaml.utils.splits.sun.trainval):
            # since the proposed split names are not separated by folders, we have to manually extract folders...
            alphabetical = class_name[0]
            directory_splits = class_name.split("_")
            path_proposal = self.get_true_path(
                alphabetical, directory_splits, image_folder
            )
            if path_proposal == False:
                print(alphabetical, directory_splits, image_folder)
                raise ValueError()
            image_list = glob.glob(path_proposal + "/*.jpg")
            traincrawler += [(item, idx, 0) for item in image_list]

        for idx, class_name in enumerate(ednaml.utils.splits.sun.query):
            # since the proposed split names are not separated by folders, we have to manually extract folders...
            alphabetical = class_name[0]
            directory_splits = class_name.split("_")
            path_proposal = self.get_true_path(
                alphabetical, directory_splits, image_folder
            )
            if path_proposal == False:
                raise ValueError()
            image_list = glob.glob(path_proposal + "/*.jpg")
            querycrawler += [
                (item, idx + len(ednaml.utils.splits.sun.trainval), 0)
                for item in image_list
            ]

        random.shuffle(traincrawler)
        split = 0.9
        split_idx = math.ceil(split * len(traincrawler))

        self.metadata["test"]["crawl"] = traincrawler[split_idx:]
        self.metadata["test"]["pids"] = len(ednaml.utils.splits.sun.trainval)
        self.metadata["test"]["cids"] = 1
        self.metadata["test"]["imgs"] = len(self.metadata["test"]["crawl"])

        self.metadata["query"]["crawl"] = querycrawler
        self.metadata["query"]["pids"] = len(ednaml.utils.splits.sun.query)
        self.metadata["query"]["cids"] = 1
        self.metadata["query"]["imgs"] = len(self.metadata["query"]["crawl"])

        self.metadata["train"]["crawl"] = traincrawler[:split_idx]
        self.metadata["train"]["pids"] = len(ednaml.utils.splits.sun.trainval)
        self.metadata["train"]["cids"] = 1
        self.metadata["train"]["imgs"] = len(traincrawler[:split_idx])

    def get_true_path(self, alphabetical, directory_splits, image_folder):
        full_name = "_".join(directory_splits)
        path_proposal = os.path.join(image_folder, alphabetical)
        path_proposal = os.path.join(path_proposal, full_name)
        if os.path.exists(path_proposal):
            return path_proposal
        # So the full name doesn't work. We'll try to build it piece by piece...
        path_proposal = os.path.join(image_folder, alphabetical)
        directory_proposal = ""
        for idx, directory in enumerate(directory_splits):
            # Build the path with the current proposal
            if directory_proposal == "":
                directory_proposal = "_".join([] + [directory])
            else:
                directory_proposal = "_".join([directory_proposal] + [directory])
            proposal = os.path.join(path_proposal, directory_proposal)
            if os.path.exists(proposal):
                path_proposal = proposal
                # reset directory proposal
                directory_proposal = ""
            # If path does not exist, we try by appending the next directory split to the directory proposal...
        if not os.path.exists(path_proposal):
            return False
        return path_proposal
