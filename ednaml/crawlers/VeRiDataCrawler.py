from collections import defaultdict
import os
import re
import glob
import xml.etree.ElementTree as ET


class VeRiDataCrawler:
    """ Data crawler for the VeRi-776 dataset

    The VeRiDataCrawler crawls the VeRi-776 folder to populate training, query, and gallery sets with images and their respective Pid and Cid.

    Args:
      data_folder (str): Name of the VeRi folder. Default: "VeRi"
      train_folder (str): Folder inside data_folder with training images. Default: "image_train"
      test_folder (str): Folder inside data_folder with testing/gallery images. Default: "image_test"
      query_folder (str): Folder inside data_folder with query images. Default: "image_query"
    
    Kwargs:
      logger: Instance of Logging object

    Attributes:
      metadata (dict): Contains image paths, Pid, and Cid of training, testing, and query sets
        train (dict): contains image paths, Pid, and Cid
          crawl (list): List of tuples. Each tuple is a 5-tuple of (image_path, PID, CID, color, type)
          pid (int): Number of Pid in training set
          cid (int): Number of Cid in training set
          imgs (int): Number of images in training set
        test (dict): contains image paths, Pid, and Cid
          crawl (list): List of tuples. Each tuple is a 5-tuple of (image_path, PID, CID, color, type)
          pid (int): Number of Pid in testing set
          cid (int): Number of Cid in testing set
          imgs (int): Number of images in testing set
        query (dict): contains image paths, Pid, and Cid
          crawl (list): List of tuples. Each tuple is a 5-tuple of (image_path, PID, CID, color, type)
          pid (int): Number of Pid in query set
          cid (int): Number of Cid in query set
          imgs (int): Number of images in query set

    Methods:
      crawl(): Populate self.metadata
      __verify(folder): Check if folder exists

    """

    def __init__(
        self,
        data_folder="VeRi",
        train_folder="image_train",
        test_folder="image_test",
        query_folder="image_query",
        **kwargs
    ):
        self.metadata = {}

        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)
        self.test_folder = os.path.join(self.data_folder, test_folder)
        self.query_folder = os.path.join(self.data_folder, query_folder)
        self.tracks_file = os.path.join(self.data_folder, "test_track.txt")
        self.testlabel = os.path.join(self.data_folder, "test_label.xml")
        self.trainlabel = os.path.join(self.data_folder, "train_label.xml")

        self.logger = kwargs.get("logger")

        self.__verify(self.data_folder)
        self.__verify(self.train_folder)
        self.__verify(self.test_folder)
        self.__verify(self.query_folder)

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
        self.typedict = defaultdict(lambda: -1)
        with open(self.trainlabel, "r") as rfile:
            traintree = ET.fromstring(rfile.read())
        # traintree = ET.parse(self.trainlabel, parser=ET.XMLParser(encoding='gb2312'))

        for item in traintree[0]:
            vid = item.get("vehicleID")
            colorid = item.get("colorID")
            typeid = item.get("typeID")
            self.colordict[int(vid)] = int(colorid) - 1
            self.typedict[int(vid)] = int(typeid) - 1
        # traintree = ET.parse(self.testlabel, parser=ET.XMLParser(encoding='gb2312'))
        with open(self.testlabel, "r") as rfile:
            traintree = ET.fromstring(rfile.read())
        # trainroot = traintree.find("Items")
        for item in traintree[0]:
            vid = item.get("vehicleID")
            colorid = item.get("colorID")
            typeid = item.get("typeID")
            self.colordict[int(vid)] = int(colorid) - 1
            self.typedict[int(vid)] = int(typeid) - 1
        del traintree

        self.classes = {}
        self.classes["color"] = 10
        self.classes["type"] = 9

        (
            self.metadata["train"],
            self.metadata["test"],
            self.metadata["query"],
            self.metadata["track"],
        ) = ({}, {}, {}, {})
        (
            self.metadata["train"]["crawl"],
            self.metadata["train"]["pid"],
            self.metadata["train"]["cid"],
            self.metadata["train"]["imgs"],
        ) = self.__crawl(self.train_folder, reset_labels=True)
        (
            self.metadata["test"]["crawl"],
            self.metadata["test"]["pid"],
            self.metadata["test"]["cid"],
            self.metadata["test"]["imgs"],
        ) = self.__crawl(self.test_folder)
        (
            self.metadata["query"]["crawl"],
            self.metadata["query"]["pid"],
            self.metadata["query"]["cid"],
            self.metadata["query"]["imgs"],
        ) = self.__crawl(self.query_folder)

        (
            self.metadata["track"]["crawl"],
            self.metadata["track"]["pid"],
            self.metadata["track"]["cid"],
            self.metadata["track"]["imgs"],
            self.metadata["track"]["dict"],
            self.metadata["track"]["info"],
        ) = self.__crawltracks(self.test_folder)

        # Extras for colabel...
        self.metadata["val"], self.metadata["full"] = {}, {}
        self.metadata["val"]["crawl"], self.metadata["full"]["crawl"] = [], []
        for meta in ["train", "test", "val", "full"]:
            self.metadata[meta]["imgs"] = len(self.metadata[meta]["crawl"])
            self.metadata[meta]["classes"] = {}
            self.metadata[meta]["classes"]["color"] = 10
            self.metadata[meta]["classes"]["type"] = 9
            self.metadata[meta]["classes"]["pid"] = self.metadata["train"]["pid"]
            self.metadata[meta]["classes"]["cid"] = self.metadata["train"]["cid"]

        self.logger.info(
            "Train\tPID: {:6d}\tCID: {:6d}\tIMGS: {:8d}".format(
                self.metadata["train"]["pid"],
                self.metadata["train"]["cid"],
                self.metadata["train"]["imgs"],
            )
        )
        self.logger.info(
            "Test \tPID: {:6d}\tCID: {:6d}\tIMGS: {:8d}".format(
                self.metadata["test"]["pid"],
                self.metadata["test"]["cid"],
                self.metadata["test"]["imgs"],
            )
        )
        self.logger.info(
            "Query\tPID: {:6d}\tCID: {:6d}\tIMGS: {:8d}".format(
                self.metadata["query"]["pid"],
                self.metadata["query"]["cid"],
                self.metadata["query"]["imgs"],
            )
        )
        self.logger.info(
            "Tracks\tPID: {:6d}\tCID: {:6d}\Tracks: {:8d}".format(
                self.metadata["track"]["pid"],
                self.metadata["track"]["cid"],
                self.metadata["track"]["imgs"],
            )
        )

    def __crawl(self, folder, reset_labels=False):
        imgs = glob.glob(os.path.join(folder, "*.jpg"))
        _re = re.compile(r"([\d]+)_[a-z]([\d]+)")
        pid_labeler = 0
        pid_tracker, cid_tracker = {}, {}
        crawler = []
        pid_counter, cid_counter, img_counter = 0, 0, 0
        for img in imgs:
            pid, cid = map(int, _re.search(img).groups())  # _re.search lol
            if pid < 0:
                continue  # ignore junk
            if cid < 0:
                continue  # ignore junk
            if pid not in pid_tracker:
                pid_tracker[pid] = pid_labeler if reset_labels else pid
                pid_labeler += 1
            if cid not in cid_tracker:
                cid_tracker[cid] = cid - 1
            crawler.append(
                (
                    img,
                    pid_tracker[pid],
                    cid - 1,
                    self.colordict[pid],
                    self.typedict[pid],
                )
            )  # cid start at 1 in data
        return crawler, len(pid_tracker), len(cid_tracker), len(crawler)

    def __crawltracks(self, folder, reset_labels=False):

        _re = re.compile(r"([\d]+)_[a-z]([\d]+)")
        pid_labeler = 0
        pid_tracker, cid_tracker = {}, {}
        crawler = []
        pid_counter, cid_counter, img_counter = 0, 0, 0
        track_dict, track_info = {}, {}
        track_idx = 0
        with open(self.tracks_file, "r") as tracks_file_reader:
            for line in tracks_file_reader:
                # each line is a track...
                track_list = line.strip().split(" ")
                track_index = track_list[0]
                track_images = track_list[1:]
                track_images = [os.path.join(folder, item) for item in track_images]
                pid, cid = map(
                    int, _re.search(track_images[0]).groups()
                )  # _re.search lol
                if pid < 0:
                    continue  # ignore junk
                if cid < 0:
                    continue  # ignore junk
                if pid not in pid_tracker:
                    pid_tracker[pid] = pid_labeler if reset_labels else pid
                    pid_labeler += 1
                crawler.append(
                    (
                        track_images,
                        pid_tracker[pid],
                        cid - 1,
                        self.colordict[pid],
                        self.typedict[pid],
                    )
                )  # cid start at 1 in data
                # if len(crawler) == 1650:
                for img in track_images:
                    track_dict[img] = track_idx
                track_info[track_idx] = {"pid": pid_tracker[pid], "cid": cid - 1}
                track_idx += 1
        return (
            crawler,
            len(pid_tracker),
            len(cid_tracker),
            len(crawler),
            track_dict,
            track_info,
        )
