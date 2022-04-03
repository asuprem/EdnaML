import os
import random, math


class Cars196DataCrawler:
    def __init__(
        self,
        data_folder="Cars196",
        train_folder="cars_train",
        test_folder="cars_test",
        query_folder="",
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

    def crawl(self,):
        self.metadata["train"], self.metadata["test"], self.metadata["query"] = (
            {},
            {},
            {},
        )
        self.__crawl(
            os.path.join(self.data_folder, "cars_train_annos.mat"), self.train_folder
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

    def __crawl(self, crawler_annotations_file, folder):
        from scipy.io import loadmat

        matfile = loadmat(crawler_annotations_file)
        # matlab file is car_train_annos.mat or car_test_annos.mat
        crawler = [
            (os.path.join(folder, str(item[5][0])), item[4][0][0] - 1, 0)
            for item in matfile["annotations"][0]
        ]
        # crawler is a list of 3-tuples. Length of N=number of images.
        # Each tuple is (path/to/image, PID, CID)   PID --> class (from 0-195), CID --> 0
        # CID is a holdover from other crawlers used for re-id task, where cid, or camera-id is required. To maintain compatibility with SequencedGenerator (which expects CID) until I write a generator for Cars196
        # PID is similarity from person-reid, where PID stands for person ID. In this case, it is a unique class

        train_crawler = [item for item in crawler if item[1] in range(0, 98)]
        test_crawler = [item for item in crawler if item[1] in range(98, 196)]

        random.shuffle(train_crawler)
        split = 0.7
        split_idx = math.ceil(split * len(train_crawler))

        self.metadata["test"]["crawl"] = train_crawler[split_idx:]
        self.metadata["test"]["pids"] = 98
        self.metadata["test"]["cids"] = 1
        self.metadata["test"]["imgs"] = len(self.metadata["test"]["crawl"])
        train_crawler = train_crawler[:split_idx]

        self.metadata["query"]["crawl"] = test_crawler
        self.metadata["query"]["pids"] = 98
        self.metadata["query"]["cids"] = 1
        self.metadata["query"]["imgs"] = len(self.metadata["query"]["crawl"])

        self.metadata["train"]["crawl"] = train_crawler[:split_idx]
        self.metadata["train"]["pids"] = 98
        self.metadata["train"]["cids"] = 1
        self.metadata["train"]["imgs"] = len(train_crawler)
