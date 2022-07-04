import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict
import random
import os.path as osp
import numpy as np

class TDataSet(TorchDataset):
  def __init__(self,dataset, transform):
    self.dataset = dataset
    self.transform = transform
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,idx):
    datatuple = self.dataset[idx]
    img_arr = self.transform(self.load(datatuple[0]))
    return img_arr, datatuple[1], datatuple[2], datatuple[0]
  
  def load(self,img):
    if not osp.exists(img):
      raise IOError("{img} does not exist in path".format(img=img))
    img_load = Image.open(img).convert('RGB')
    return img_load

class TSampler(Sampler):
  """ Triplet sampler """
  def __init__(self, dataset, batch_size, instance):
    self.dataset = dataset
    self.batch_size=batch_size
    self.instance = instance
    self.unique_ids = self.batch_size // self.instance
    self.indices = defaultdict(list)
    self.pids = set()
    # Record the indices for each pid in a self.indices    pid->[list of indices] from the crawled dataset...
    # Also record the pids in self.pids
    for idx, datatuple in enumerate(self.dataset):  # (img, pid, cid)
      self.indices[datatuple[1]].append(idx)
      self.pids.add(datatuple[1])
    self.pids = list(self.pids)
    self.batch = 0
    for pid in self.pids:
      num_ids = len(self.indices[pid])  # number of indices in pid, then upscaled to self.instance. 
      # Basically, self-batch how many <self.instances>-groups of same indices are there.
      # instance=3. 
      # If we see a [1,1],            we upscale   to [1,1,1]. 
      # If we see a [1,1,1,1],        we downscale to [1,1,1]. 
      # If we see a [1,1,1,1,1,1,1],  we downscale to [1,1,1,1,1,1]
      num_ids = self.instance if num_ids < self.instance else num_ids
      self.batch += num_ids - num_ids % self.instance
  
  def __iter__(self):
    batch_idx = defaultdict(list)
    for pid in self.pids:
      ids = [item for item in self.indices[pid]]  # copy of indices
      if len(ids) < self.instance:  
        ids = np.random.choice(ids, size=self.instance, replace=True) # upscaling, here...
      random.shuffle(ids) # shuffle the indices
      batch, batch_counter = [], 0
      for idx in ids:
        batch.append(idx) # add the indices to batch
        batch_counter += 1
        if len(batch) == self.instance: # Once we have a <instance> sized batch of idx, add them to the batchidx   pid -> [<instance> list of ids]
          batch_idx[pid].append(batch)
          batch = []
    _pids, r_pids = [item for item in self.pids], []  # all the pids, as a copy, again
    # Optimize this???
    to_remove = {}
    pid_len = len(_pids)
    # This inequality is because each round, we are taking <unique-ids> number of pids, and 1 image from each pid. Since we have "upscaled" the list of images in each pid, at some point,
    # we will arrive at <unique_ids> pids in the the list of pids with images still there. When we take 1 more image out, there will be 0 pids in there after they are removed...
    while pid_len >= self.unique_ids:   # so, instance is num per each pid. This means unique_ids is number of <instance>-sized unique ids...
      sampled = random.sample(_pids, self.unique_ids) # get a random sample of pids, the size of unique ids
      for pid in sampled:
        batch = batch_idx[pid].pop(0)   #aaaand, the magic happens here. Get 1 from pid -> [<instance> list of ids]
        r_pids.extend(batch)            # add that index to the r_pids
        if len(batch_idx[pid]) == 0:    # if there are no more images of this pid in the pid-to-idx map, mark this pid for removal in the current batch creation
          to_remove[pid] = 1

      _pids = [item for item in _pids if item not in to_remove] # update the pids by removing the marked pid
      pid_len = len(_pids)  # update the length
      to_remove = {}

    self.__len = len(r_pids)
    return iter(r_pids)   # This ensures we return an interable over r_pids...
  
  def __len__(self):
    return self.__len
class SequencedGenerator:
  def __init__(self,gpus, i_shape = (208,208), normalization_mean = 0.5, normalization_std = 0.5, normalization_scale = 1./255., h_flip = 0.5, t_crop = True, rea = True, **kwargs):
    """ Data generator for training and testing. Works with the VeriDataCrawler. Should work with any crawler working on VeRi-like data. Not yet tested with VehicleID. Only  use with VeRi.

    Generates batches of batch size CONFIG.TRANSFORMATION.BATCH_SIZE, with CONFIG.TRANSFORMATION.INSTANCE unique ids. So if BATCH_SIZE=36 and INSTANCE=6, then generate batch of 36 images, with 6 identities, 6 image per identity. See arguments of setup function for INSTANCE.

    Args:
      gpus (int): Number of GPUs
      i_shape (int, int): 2D Image shape
      normalization_mean (float): Value to pass as mean normalization parameter to pytorch Normalization
      normalization_std (float): Value to pass as std normalization parameter to pytorch Normalization
      normalization_scale (float): Value to pass as scale normalization parameter. Not used.
      h_flip (float): Probability of horizontal flip for image
      t_crop (bool): Whether to include random cropping
      rea (bool): Whether to include random erasing augmentation (at 0.5 prob)
    
    """
    self.gpus = gpus
    
    transformer_primitive = []
    
    transformer_primitive.append(T.Resize(size=i_shape))
    if h_flip > 0:
      transformer_primitive.append(T.RandomHorizontalFlip(p=h_flip))
    if t_crop:
      transformer_primitive.append(T.RandomCrop(size=i_shape))
    transformer_primitive.append(T.ToTensor())
    transformer_primitive.append(T.Normalize(mean=normalization_mean, std=normalization_std))
    if rea:
      transformer_primitive.append(T.RandomErasing(p=0.5, scale=(0.02, 0.4), value = kwargs.get('rea_value', 0)))
    self.transformer = T.Compose(transformer_primitive)

  def setup(self,datacrawler, mode='train', batch_size=32, instance = 8, workers = 8):
    """ Setup the data generator.

    Args:
      workers (int): Number of workers to use during data retrieval/loading
      datacrawler (VeRiDataCrawler): A DataCrawler object that has crawled the data directory
      mode (str): One of 'train', 'test', 'query'. 
    """
    if datacrawler is None:
      raise ValueError("Must pass DataCrawler instance. Passed `None`")
    self.workers = workers * self.gpus

    if mode == "train":
      self.__dataset = TDataSet(datacrawler.metadata[mode]["crawl"], self.transformer)
    elif mode == "test":
      # For testing, we combine images in the query and testing set to generate batches
      self.__dataset = TDataSet(datacrawler.metadata["query"]["crawl"] + datacrawler.metadata[mode]["crawl"], self.transformer)
    else:
      raise NotImplementedError()
    
    if mode == "train":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        sampler = TSampler(datacrawler.metadata[mode]["crawl"], batch_size=batch_size*self.gpus, instance=instance*self.gpus), \
                                        num_workers=self.workers, collate_fn=self.collate_simple)
      self.num_entities = datacrawler.metadata[mode]["pids"]
    elif mode == "test":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        shuffle = False, 
                                        num_workers=self.workers, collate_fn=self.collate_with_camera)
      self.num_entities = len(datacrawler.metadata["query"]["crawl"])
    else:
      raise NotImplementedError()
    
  def collate_simple(self,batch):
    img, pid, _, _ = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    return torch.stack(img, dim=0), pid
  def collate_with_camera(self,batch):
    img, pid, cid, path = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    cid = torch.tensor(cid, dtype=torch.int64)
    return torch.stack(img, dim=0), pid, cid, path
