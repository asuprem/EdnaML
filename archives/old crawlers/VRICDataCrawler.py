import os

from ednaml.crawlers import Crawler


class VRICDataCrawler(Crawler):
    """Data crawler for the VRIC (Vehicle Re-identification in Context) dataset

    The VRIC crawls the VRIC data folder to populate training, query, and gallery sets with images and their respective PIDs and CIDs. It uses the provided list of training, query, and gallery images to populate the crawler metadata.

    Args:
      data_folder (str): Name of the VeRi folder. Default: "VRIC"
      train_folder (str): Folder inside data_folder with training images. Default: "train_images"
      test_folder (str): Folder inside data_folder with testing/gallery images. Default: "gallery_images"
      query_folder (str): Folder inside data_folder with query images. Default: "probe_images"

    Kwargs:
      logger: Instance of Logging object

    Attributes:
      metadata (dict): Contains image paths, PIDs, and CIDs of training, testing, and query sets
        train (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in training set
          cids (int): Number of CIDs in training set
          imgs (int): Number of images in training set
        test (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in testing set
          cids (int): Number of CIDs in testing set
          imgs (int): Number of images in testing set
        query (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in query set
          cids (int): Number of CIDs in query set
          imgs (int): Number of images in query set

    Methods:
      crawl(): Populate self.metadata
      __verify(folder): Check if folder exists

    """

    def __init__(
        self,
        data_folder="VRIC",
        train_folder="train_images",
        test_folder="gallery_images",
        query_folder="probe_images",
        **kwargs
    ):
        self.metadata = {}

        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)
        self.test_folder = os.path.join(self.data_folder, test_folder)
        self.query_folder = os.path.join(self.data_folder, query_folder)

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

    def crawl(
        self,
    ):
        # We will build the details using the text files
        (
            self.metadata["train"],
            self.metadata["test"],
            self.metadata["query"],
        ) = (
            {},
            {},
            {},
        )
        (
            self.metadata["train"]["crawl"],
            self.metadata["train"]["pids"],
            self.metadata["train"]["cids"],
            self.metadata["train"]["imgs"],
        ) = self.__crawl(
            os.path.join(self.data_folder, "vric_train.txt"),
            self.train_folder,
            reset_labels=True,
        )
        (
            self.metadata["test"]["crawl"],
            self.metadata["test"]["pids"],
            self.metadata["test"]["cids"],
            self.metadata["test"]["imgs"],
        ) = self.__crawl(
            os.path.join(self.data_folder, "vric_gallery.txt"),
            self.test_folder,
            reset_labels=True,
        )
        (
            self.metadata["query"]["crawl"],
            self.metadata["query"]["pids"],
            self.metadata["query"]["cids"],
            self.metadata["query"]["imgs"],
        ) = self.__crawl(
            os.path.join(self.data_folder, "vric_probe.txt"),
            self.query_folder,
            reset_labels=True,
        )

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

    def __crawl(self, file_to_open, folder, reset_labels=False):
        pid_labeler = 0
        pid_tracker, cid_tracker = {}, {}
        crawler = []
        pid_counter, cid_counter, img_counter = 0, 0, 0

        with open(file_to_open, "r") as file_:
            for line in file_:
                path, pid, cid = line.strip().split(" ")
                pid = int(pid)
                cid = int(cid)
                path = os.path.join(folder, path)
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
                    (path, pid_tracker[pid], cid - 1)
                )  # cids start at 1 in data
            return crawler, len(pid_tracker), len(cid_tracker), len(crawler)
