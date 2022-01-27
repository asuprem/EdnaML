import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

import torchvision.transforms as T

class CoLabelDataset(TorchDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(read_image(self.dataset[idx][0])), self.dataset[idx][0]

    

class CoLabelGenerator:
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

    
    def setup(self, datacrawler, mode='train', batch_size=32, instance = 8, workers = 8):
        """ Setup the data generator.

        Args:
            workers (int): Number of workers to use during data retrieval/loading
            datacrawler (VeRiDataCrawler): A DataCrawler object that has crawled the data directory
            mode (str): One of 'train', 'test', 'validation???'. 
        """

        if datacrawler is None:
            raise ValueError("Must pass DataCrawler instance. Passed `None`")
        self.workers = workers * self.gpus

        if mode == "train":
            self.__dataset = CoLabelDataset(datacrawler.metadata[mode]["crawl"], self.transformer)
        elif mode == "test":
            # For testing, we combine images in the query and testing set to generate batches
            self.__dataset = CoLabelDataset(datacrawler.metadata["val"]["crawl"] + datacrawler.metadata[mode]["crawl"], self.transformer)
        else:
            raise NotImplementedError()

        if mode == "train":
            self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, shuffle=True,\
                                                num_workers=self.workers, collate_fn=self.collate_simple)
            self.num_entities = datacrawler.metadata[mode]["classes"]
        elif mode == "test":
            self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, shuffle=True,\
                                                num_workers=self.workers, collate_fn=self.collate_simple)
            self.num_entities = datacrawler.metadata[mode]["classes"]
        else:
            raise NotImplementedError()

    def collate_simple(self,batch):
        img, class_id, = zip(*batch)
        class_id = torch.tensor(class_id, dtype=torch.int64)
        return torch.stack(img, dim=0), class_id