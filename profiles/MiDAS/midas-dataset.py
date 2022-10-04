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

    <datasetname>-<labelsubset>-<split>.csv

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
          crawl.append([row[textidx], int(row[labelidx]), datasetidx])
    return crawl



import torch
from torch.utils.data import TensorDataset
class MiDASDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, mode, transform=None, **kwargs):
    self.dataset = dataset  # list of tuples (text, label, datasetlabel)
    self.cache = kwargs.get("cache", False)
    self.memcache = kwargs.get("memcache", False)
    self.tokenizer = kwargs.get("tokenizer")
    self.maxlen = kwargs.get("maxlen")
    self.mlm = kwargs.get("mlm_probability", 0.2)
    self.masking = kwargs.get("masking", mode=="train")

    self.getcount = 0
    self.refresh_flag = len(dataset)

    if self.cache and self.memcache:
      raise ValueError("Use only one cache type")

    self.getter = self.uncachedget
    if self.cache:
      raise NotImplementedError()
      self.getter = self.diskget
    if self.memcache:
      self.getter = self.memget

    # if cache, then we will cache transformations to disk, then load them
    # if memcache, then we will cache to memory and load them

    if self.cache or self.memcache:
      # The actual cache-ing
      self.input_length_cache = []
      self.convert_to_features(self.dataset, self.tokenizer, maxlen=self.maxlen)
      self.memcached_dataset = self.refresh_mask_ids()
      

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.getter(idx)

  def uncachedget(self, idx):
    return 

  def memget(self, idx):
    self.getcount+=1
    if self.getcount > self.refresh_flag and self.masking:
      self.memcached_dataset = self.refresh_mask_ids()
      self.getcount = 0
    return self.memcached_dataset[idx]

  def refresh_mask_ids(self):
    print("Refreshing mask ids")
    if self.masking:
      self.mask_ids = self.build_mask_ids(self.input_length_cache)
      
      all_attention_mask = self.all_attention_mask.clone()
      all_masklm = self.all_masklm.clone()

      for idx in range(self.refresh_flag):
        all_attention_mask[idx][self.mask_ids[idx]] = 0 # Set the masking words to 0, so we do not attend to it during prediction
        all_masklm[idx][self.mask_ids[idx]] = self.all_input_ids[idx][self.mask_ids[idx]] # Set the masking labels for these to the actual word index from all_input_ids

      return TensorDataset(self.all_input_ids, all_attention_mask, self.all_token_type_ids, all_masklm, self.all_lens, self.all_labels, self.all_datalabels)
    else:
      return TensorDataset(self.all_input_ids, self.all_attention_mask, self.all_token_type_ids, self.all_masklm, self.all_lens, self.all_labels, self.all_datalabels)

  def build_mask_ids(self, input_length_cache):
    # for each element, we get a set of indices that are randomly selected...
    # Also, -2 and +1 take care of [cls] and [sep] not being masked
    return [(torch.randperm(inplength-2)+1)[:int(inplength*self.mlm)]  for inplength in input_length_cache]


  def convert_to_features(self, dataset, tokenizer, maxlen):
    features = []
    self.input_length_cache = []
    for idx, sample in enumerate(dataset):
      tokens = self.tokenizer.tokenize(sample[0])
      if len(tokens) > maxlen - 2:
        tokens = tokens[0:(maxlen - 2)]

      finaltokens = ["[CLS]"]
      token_type_ids = [0]
      for token in tokens:
        finaltokens.append(token)
        token_type_ids.append(0)
      finaltokens.append("[SEP]")
      token_type_ids.append(0)

      input_ids = self.tokenizer.convert_tokens_to_ids(finaltokens)
      attention_mask = [1]*len(input_ids)
      input_len = len(input_ids)
      self.input_length_cache.append(len(input_ids))
      while len(input_ids) < maxlen:
        input_ids.append(0)
        attention_mask.append(0)
        token_type_ids.append(0) 

      assert len(input_ids) == maxlen
      assert len(attention_mask) == maxlen
      assert len(token_type_ids) == maxlen
      
      features.append(
          (input_ids, attention_mask, token_type_ids, input_len, sample[1], sample[2])
      )

    self.all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    self.all_attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    self.all_token_type_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    self.all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
    self.all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
    self.all_datalabels = torch.tensor([f[5] for f in features], dtype=torch.long)
    self.all_masklm = -1*torch.ones(self.all_input_ids.shape, dtype=torch.long)



from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.generators import TextGenerator

@edna.register_generator
class MiDASGenerator(TextGenerator):
  # input includes tokenizer for build...

  # Set it up such that, given a crawler, create a dataset from it.
  # Then create a cached batch
  # From cached batch, yield batches until it is empty

  def build_transforms(self, transform, mode, **kwargs):  #<-- generator kwargs:
    from ednaml.utils import locate_class
    tokenizer = kwargs.get("tokenizer", "AlbertFullTokenizer")
    self.tokenizer = locate_class(package="ednaml", subpackage="utils", classpackage=tokenizer, classfile="tokenizers")
    self.tokenizer = self.tokenizer(**kwargs) # vocab_file, do_lower_case, spm_model_file

  
  def buildDataset(self, datacrawler, mode, transform, **kwargs): #<-- dataset args:
    return MiDASDataset(datacrawler.metadata[mode]["crawl"], mode, tokenizer = self.tokenizer, **kwargs) # needs maxlen, memcache, mlm_probability

  def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size*self.gpus,
                                        shuffle=True, num_workers = self.workers, 
                                       collate_fn=self.collate_fn)

  def getNumEntities(self, crawler, mode, **kwargs):  #<-- dataset args
    label_dict = {
        item: {"classes": crawler.metadata[mode]["classes"][item]}
        for item in kwargs.get("classificationclass", ["color"])
    }
    return LabelMetadata(label_dict=label_dict)

  def collate_fn(self, batch):
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_lens, all_labels, all_datalabels  = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_masklm = all_masklm[:, :max_len]

    return all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels, all_datalabels