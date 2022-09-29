import ednaml, torch, csv, ctypes as ct, os
import ednaml.core.decorators as edna
from ednaml.crawlers import Crawler
from ednaml.utils.web import download

@edna.register_crawler
class PolitifactCrawler(Crawler):
    def __init__(self, logger = None, split = 0.9, true_url = None, fake_url = None, true_file = "politifact_real.csv", fake_file = "politifact_fake.csv"):
        csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
        self.true_url = true_url
        self.fake_url = fake_url
        
        if self.fake_url is None:
            self.fake_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv"
        if self.true_url is None:
            self.true_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv"
        
        self.fake_file = fake_file
        self.true_file = true_file
        
        if not os.path.exists(self.fake_file):
            download(self.fake_file, self.fake_url)
        if not os.path.exists(self.true_file):
            download(self.true_file, self.true_url)

        all_fakes = []
        all_true = []


        with open(self.fake_file, "r") as ffile:
            fobj = csv.reader(ffile, delimiter=",")
            header = next(fobj)
            for row in fobj:
                all_fakes.append(
                    (row[2], 0)
                )
        
        with open(self.true_file, "r") as ffile:
            fobj = csv.reader(ffile, delimiter=",")
            header = next(fobj)
            for row in fobj:
                all_true.append(
                    (row[2], 1)
                )

        import random
        random.seed(345345)
        random.shuffle(all_fakes)
        random.shuffle(all_true)
        
        self.metadata = {}
        self.metadata["train"] = {}
        self.metadata["test"] = {}
        self.metadata["val"] = {}
        self.metadata["train"]["crawl"] = []
        self.metadata["test"]["crawl"] = []
        self.metadata["val"]["crawl"] = []
        

        fsplit = int(len(all_fakes)*split)
        tsplit = int(len(all_true)*split)
        self.metadata["train"]["crawl"] += all_fakes[:fsplit]
        self.metadata["train"]["crawl"] += all_true[:tsplit]
        self.metadata["test"]["crawl"] += all_fakes[fsplit:]
        self.metadata["test"]["crawl"] += all_true[tsplit:]

        random.shuffle(self.metadata["train"]["crawl"])
        random.shuffle(self.metadata["test"]["crawl"])

        self.classes = {}
        self.classes["fnews"] = 2
        
        self.metadata["train"]["classes"] = self.classes
        self.metadata["test"]["classes"] = self.classes

        