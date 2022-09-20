import ednaml
import csv, glob, os
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna

@edna.register_crawler
class MiDASCrawler(Crawler):
  def __init__(self, logger = None, data_folder="Data", include=["cmu_miscov19", "covid19_fn_title", "kagglefn_short"]):
    """Crawls the Data folder with all datasets already extracted to their individual folders.
    Assumes specific file construction: files are separated into splits 
    (train, test, and val), with label subsets fake and true, with naming convention:


    Crawl yields lists with tuples of:
    <text>, <shares>, <likes>, <label>, <label...s>

    """
    logger.info("Crawling %s for %s"%(data_folder, str(include)))
    
    self.classes = {}
    self.classes["fnews"] = 2
    self.metadata = {}

    self.data_folder = data_folder
    datasets_folders = glob.glob(os.path.join(data_folder, "*"))
    datasets_folders = [item for item in datasets_folders if os.path.basename(item) in include]
    self.classes["dataset"] = len(datasets_folders)


    self.metadata["train"] = {}
    self.metadata["test"] = {}
    self.metadata["val"] = {}
    self.metadata["train"]["crawl"] = []
    self.metadata["test"]["crawl"] = []
    self.metadata["val"]["crawl"] = []
    for idx, folder in enumerate(datasets_folders):
      ftrain = [os.path.join(folder, "-".join([os.path.basename(folder), subset, "train.csv"])) for subset in ["fake", "true"]]
      ftest = [os.path.join(folder, "-".join([os.path.basename(folder), subset, "test.csv"])) for subset in ["fake", "true"]]
      fval = [os.path.join(folder, "-".join([os.path.basename(folder), subset, "val.csv"])) for subset in ["fake", "true"]]

      # train:
      self.metadata["train"]["crawl"] += self.getTextAndLabels(ftrain, idx)
      self.metadata["test"]["crawl"] += self.getTextAndLabels(ftest, idx)
      self.metadata["val"]["crawl"] += self.getTextAndLabels(fval, idx)

    self.metadata["train"]["classes"] = self.classes
    self.metadata["test"]["classes"] = self.classes
    self.metadata["val"]["classes"] = self.classes

  def getTextAndLabels(self, listOfFiles, datasetidx):
    crawl = []
    for file in listOfFiles:
      # For each file, open as csv. Extract the relevant columns ("text", "label")
      with open(file, "r") as ofile:
        csvread = csv.reader(ofile)

        header = next(csvread)
        textidx = [idx for idx,item in enumerate(header) if item == "text"][0]
        labelidx = [idx for idx,item in enumerate(header) if item == "label"][0]

        for row in csvread:
          crawl.append([row[textidx], 0,0,int(row[labelidx]), datasetidx])
    return crawl
