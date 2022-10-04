import ednaml, torch, os, csv, os, json, glob, click
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna


@edna.register_crawler
class NELACrawler(Crawler):
  def __init__(self, logger = None, data_folder="Data", sub_folder="nela-covid-2020"):
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
    self.data_folder = os.path.join(data_folder , sub_folder)
    labelfile = os.path.join(self.data_folder, "labels.csv")
    self.data_folder = os.path.join(self.data_folder, sub_folder) # because there is nested
    ndata = "newsdata"
    tweet = "tweet"
    
    self.newsdata = os.path.join(self.data_folder, ndata)
    self.tweetdata = os.path.join(self.data_folder, tweet)

    # obtain source labels
    sourcelabels = {}
    with open(labelfile, "r") as lfile:
      labelobj = csv.reader(lfile)
      header = next(labelobj)
      for row in labelobj:
        sourcelabels[row[0]] = int(row[1])

    # set up content metadata
    self.metadata = {}
    self.metadata["train"] = {}
    self.metadata["test"] = {}
    self.metadata["val"] = {}
    self.metadata["train"]["crawl"] = []
    self.metadata["test"]["crawl"] = []
    self.metadata["val"]["crawl"] = []
    
    
    # We don't need the tweet-data because it is the same as the newsdata
    #with open(os.path.join(self.tweetdata, "tweet.json")) as obj:
    #  self.tweetdata = json.load(obj)


    # for each file in newsdata: obtain the titles, and save in label propagated. We willa dd to metadata later...
    newsdataitems = glob.glob(os.path.join(self.newsdata, "*.json"))
    labelsets = {0:[],1:[],2:[]}
    for newsdatafile in newsdataitems:
      newssource = os.path.splitext(os.path.basename(newsdatafile))[0]
      label = sourcelabels.get(newssource,1)
      njson = None
      with open(newsdatafile) as jsonobj:
        njson = json.load(jsonobj)
      labelsets[label] += [item["title"] for item in njson]

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
