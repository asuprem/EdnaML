from collections import defaultdict
import os
import random

# Tuple:
# (imgpath, pid, cid, color, model)


class VehicleIDDataCrawler:
    def __init__(
        self,
        data_folder="VehicleID",
        train_folder="image",
        test_folder="",
        query_folder="",
        **kwargs
    ):
        self.metadata = {}

        self.data_folder = data_folder
        self.image_folder = os.path.join(self.data_folder, train_folder)
        self.attribute_folder = os.path.join(
            self.data_folder, kwargs.get("attribute_folder", "attribute")
        )

        self.train_list = "train_list.txt"
        self.query_list = kwargs.get("test_list", "test_list_2400") + ".txt"

        list_folder = os.path.join(self.data_folder, "train_test_split")
        # The train list is in VehicleID/train_test_split/train_list.txt
        # The gallery/query list is in VehicleID/train_test_split/test_list_13164.txt
        self.train_list = os.path.join(list_folder, self.train_list)
        self.query_list = os.path.join(list_folder, self.query_list)

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

        self.colordict = defaultdict(lambda: -1)
        with open(os.path.join(self.attribute_folder, "color_attr.txt"), "r") as cfile:
            for line in cfile:
                line = line.strip().split(" ")
                self.colordict[int(line[0])] = int(
                    line[1]
                )  # colordict[vehicleid]=colorid
        self.modeldict = defaultdict(lambda: -1)
        with open(os.path.join(self.attribute_folder, "model_attr.txt"), "r") as cfile:
            for line in cfile:
                line = line.strip().split(" ")
                self.modeldict[int(line[0])] = int(
                    line[1]
                )  # modeldict[vehicleid]=modelid

        self.classes = {}
        self.classes["color"] = 7
        self.classes["model"] = 250

        self.metadata["train"], self.metadata["gallery"], self.metadata["query"] = (
            {},
            {},
            {},
        )
        (
            self.metadata["train"]["crawl"],
            self.metadata["train"]["pids"],
            self.metadata["train"]["cids"],
            self.metadata["train"]["imgs"],
        ) = self.__crawl(self.train_list, reset_labels=True)

        self.metadata["test"] = {}
        # self.metadata["test"]["crawl"] = [self.metadata["train"]["crawl"].pop(random.randrange(len(self.metadata["train"]["crawl"]))) for _ in range(int(len(self.metadata["train"]["crawl"])*0.1))]
        random.shuffle(self.metadata["train"]["crawl"])
        nshuffle = int(len(self.metadata["train"]["crawl"]) * 0.1)
        self.metadata["test"]["crawl"] = self.metadata["train"]["crawl"][:nshuffle]
        self.metadata["train"]["crawl"] = self.metadata["train"]["crawl"][nshuffle:]
        self.metadata["train"]["imgs"] = len(self.metadata["train"]["crawl"])
        self.metadata["test"]["imgs"] = len(self.metadata["test"]["crawl"])

        self.__querycrawl(self.query_list)

        # Extras for colabel...
        self.metadata["val"], self.metadata["full"] = {}, {}
        self.metadata["val"]["crawl"], self.metadata["full"]["crawl"] = [], []
        for meta in ["train", "gallery", "val", "full", "test"]:
            self.metadata[meta]["imgs"] = len(self.metadata[meta]["crawl"])
            self.metadata[meta]["classes"] = {}
            self.metadata[meta]["classes"]["color"] = 7
            self.metadata[meta]["classes"]["make"] = 250

        self.logger.info(
            "Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["train"]["pids"],
                self.metadata["train"]["cids"],
                self.metadata["train"]["imgs"],
            )
        )
        self.logger.info(
            "Gallery \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["gallery"]["pids"],
                self.metadata["gallery"]["cids"],
                self.metadata["gallery"]["imgs"],
            )
        )
        self.logger.info(
            "Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(
                self.metadata["query"]["pids"],
                self.metadata["query"]["cids"],
                self.metadata["query"]["imgs"],
            )
        )

    def __crawl(self, train_file, reset_labels=False):
        crawler = []
        pids, cids = {}, []
        pid_label = 0
        with open(train_file, "r") as train_file_reader:
            for line in train_file_reader:
                ln = line.strip().split(" ")
                img_path = ln[0] + ".jpg"
                img_path = os.path.join(self.image_folder, img_path)
                pid = int(ln[1])
                cid = 0
                if pid not in pids:
                    pids[pid] = pid_label if reset_labels else pid
                    pid_label += 1
                # pids.append(pid)
                cids.append(cid)
                crawler.append(
                    (img_path, pids[pid], cid, self.colordict[pid], self.modeldict[pid])
                )
        return crawler, len(set(pids.keys())), len(set(cids)), len(crawler)

    def __querycrawl(self, query_file, reset_labels=False):
        crawler = []
        pids, cids = {}, []
        pid_label = 0
        with open(query_file, "r") as query_file_reader:
            for line in query_file_reader:
                ln = line.strip().split(" ")
                img_path = ln[0] + ".jpg"
                img_path = os.path.join(self.image_folder, img_path)
                pid = int(ln[1])
                cid = 0
                if pid not in pids:
                    pids[pid] = pid_label if reset_labels else pid
                    pid_label += 1
                # pids.append(pid)
                cids.append(cid)
                crawler.append((img_path, pids[pid], cid))

        pid_in_gallery = {}
        self.metadata["gallery"]["crawl"], self.metadata["query"]["crawl"] = [], []

        for crawled_img in crawler:
            img_path, pid, cid = crawled_img
            # check if pid already captured. If so add to query. Else add to gallery (based on paper) (variable pid_in_gallery should be pid_in)gallery
            if pid in pid_in_gallery:
                self.metadata["query"]["crawl"].append(
                    (img_path, pid, cid, self.colordict[pid], self.modeldict[pid])
                )
            else:
                pid_in_gallery[pid] = 1
                self.metadata["gallery"]["crawl"].append(
                    (img_path, pid, cid, self.colordict[pid], self.modeldict[pid])
                )

        self.metadata["gallery"]["pids"], self.metadata["gallery"]["cids"] = (
            len(pids),
            1,
        )
        self.metadata["query"]["pids"], self.metadata["query"]["cids"] = len(pids), 1

        self.metadata["gallery"]["imgs"] = len(self.metadata["gallery"]["crawl"])
        self.metadata["query"]["imgs"] = len(self.metadata["query"]["crawl"])
