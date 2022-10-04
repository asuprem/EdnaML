import ednaml, torch, os, csv
from ednaml.crawlers import Crawler

import ednaml.core.decorators as edna

@edna.register_crawler
class FakedditCrawler(Crawler):
    def __init__(self, logger = None, data_folder="Data/Fakeddit"):
        """Crawls the Data folder with all datasets already extracted to their individual folders.
        Assumes specific file construction: files are separated into splits 
        (train, test, and val), with label subsets fake and true, with naming convention:

        Crawl yields lists with tuples of:
        <text>, <shares>, <likes>, <label>, <label...s>

        """
        logger.info("Crawling %s"%(data_folder))
        
        # set up class metadata
        self.classes = {}
        self.classes["fnews"] = 2

        # set up paths
        self.data_folder = data_folder
        self.trainfile = os.path.join(data_folder, "all_train.tsv")
        self.valfile = os.path.join(data_folder, "all_validate.tsv")
        self.testfile = os.path.join(data_folder, "all_test_public.tsv")

        # set up content metadata
        self.metadata = {}
        self.metadata["train"] = {}
        self.metadata["test"] = {}
        self.metadata["val"] = {}
        self.metadata["train"]["crawl"] = []
        self.metadata["test"]["crawl"] = []
        self.metadata["val"]["crawl"] = []
        
        # Row information for Fakeddit:
        #   -- 5:  clean text
        #   -- 13:  score (likes)
        #   -- 17:  2 way label
        #   -- -1:  6-way label
        #   -- 12: subreddit name
        with open(self.trainfile, "r") as fl:
            fobj = csv.reader(fl, delimiter="\t")
            header = next(fobj)
            for row in fobj:
                self.metadata["train"]["crawl"] += [(row[5], 0, int(row[13]), int(row[17]), int(row[-1]))]
        with open(self.valfile, "r") as fl:
            fobj = csv.reader(fl, delimiter="\t")
            header = next(fobj)
            for row in fobj:
                self.metadata["val"]["crawl"] += [(row[5], 0, int(row[13]), int(row[17]), int(row[-1]))]
        with open(self.testfile, "r") as fl:
            fobj = csv.reader(fl, delimiter="\t")
            header = next(fobj)
            for row in fobj:
                self.metadata["test"]["crawl"] += [(row[5], 0, int(row[13]), int(row[17]), int(row[-1]))]
        

        self.metadata["train"]["classes"] = self.classes
        self.metadata["test"]["classes"] = self.classes
        self.metadata["val"]["classes"] = self.classes