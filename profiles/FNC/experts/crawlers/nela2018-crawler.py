import ednaml, torch, os, csv, os, json, glob, click
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna


@edna.register_crawler
class NELA2018Crawler(Crawler):
  def __init__(self, logger = None, data_folder="Data", sub_folder="nela-gt-2018"):
    """
    Crawl yields lists with tuples of:
    <text>, <shares>, <likes>, <label>, <label...s>

    Shares and Likes are both zero, because the data does not contain these information

    we use the nela_fake, nela_true, and nela_unsure, specifically fake and true...

    we will load them into memory, shuffle them, then save them into Crawler metadata

    """
    logger.info("Crawling %s"%(data_folder))
    
    # set up class metadata
    self.classes = {}
    self.classes["fnews"] = 2
    
    # set up paths
    self.data_folder = os.path.join(data_folder , sub_folder)
    
    truefile = os.path.join(self.data_folder, "nela_true.jsonl")
    fakefile = os.path.join(self.data_folder, "nela_fake.jsonl")
    unsuerfile = os.path.join(self.data_folder, "nela_unsure.jsonl")
    
    # set up content metadata
    self.metadata = {}
    self.metadata["train"] = {}
    self.metadata["test"] = {}
    self.metadata["val"] = {}
    self.metadata["train"]["crawl"] = []
    self.metadata["test"]["crawl"] = []
    self.metadata["val"]["crawl"] = []
    
    
    #0 --> reliable; 1 --> mixed; 2 --> unreliable
    labelsets = {0:[],1:[],2:[]}
    
    with open(truefile,"r") as topen:
        for line in topen:
            jobj = json.loads(line.strip())
            labelsets[0] += [jobj["headline"]]
    with open(fakefile,"r") as topen:
        for line in topen:
            jobj = json.loads(line.strip())
            labelsets[2] += [jobj["headline"]]
    with open(unsuerfile,"r") as topen:
        for line in topen:
            jobj = json.loads(line.strip())
            labelsets[1] += [jobj["headline"]]

    # shuffle
    import random
    random.seed(75837)
    random.shuffle(labelsets[0])
    random.seed(75837)
    random.shuffle(labelsets[1])
    random.seed(75837)
    random.shuffle(labelsets[2])

    #Splits
    splits = 0.8
    reliable_train = int(len(labelsets[0])*0.8)
    unreliable_train = int(len(labelsets[2])*0.8)
    reliable_val = int(len(labelsets[0])*0.1)
    unreliable_val = int(len(labelsets[2])*0.1)
    
    # WE WILL ADJUST LABEL --> 0 is fake, 1 is true
    # Like share count is not provided...
    # Maybe we can extract it by looking up the status ID
    self.metadata["train"]["crawl"] += [(item, 0,0,1) for item in labelsets[0][:reliable_train]]
    self.metadata["train"]["crawl"] += [(item, 0,0,0) for item in labelsets[2][:unreliable_train]]
    random.shuffle(self.metadata["train"]["crawl"])

    self.metadata["val"]["crawl"] += [(item, 0,0,1) for item in labelsets[0][reliable_train:reliable_train+reliable_val]]
    self.metadata["val"]["crawl"] += [(item, 0,0,0) for item in labelsets[2][unreliable_train:unreliable_train+unreliable_val]]
    random.shuffle(self.metadata["val"]["crawl"])

    self.metadata["test"]["crawl"] += [(item, 0,0,1) for item in labelsets[0][reliable_train+reliable_val:]]
    self.metadata["test"]["crawl"] += [(item, 0,0,0) for item in labelsets[2][unreliable_train+unreliable_val:]]
    random.shuffle(self.metadata["test"]["crawl"])
      

    self.metadata["train"]["classes"] = self.classes
    self.metadata["test"]["classes"] = self.classes
    self.metadata["val"]["classes"] = self.classes
