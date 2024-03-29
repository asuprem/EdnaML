import os.path as osp
from typing import List
import torch
from PIL import Image
from PIL import ImageFile

from ednaml.utils.LabelMetadata import LabelMetadata

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from ednaml.generators import ImageGenerator


class MultiClassificationDataset(TorchDataset):
    def __init__(self, dataset, transform=None, **kwargs):
        self.dataset = dataset
        self.transform = transform
        self.pathidx = kwargs.get("pathidx", 0)
        self.annotationidx = kwargs.get(
            "annotationidx", [1]
        )  # This should be a list...
        self.num_annotations = len(self.annotationidx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # NOTE we will need to do our idx transforms here as well...
        # NOTE or leave it to be done in the crawler, and leave the generator as pristine and untouched???
        return (
            self.transform(self.load(self.dataset[idx][self.pathidx])),
            [
                self.dataset[idx][self.annotationidx[annot]]
                for annot in range(self.num_annotations)
            ],
        )  # NOTE make this return a tuple for deploy (tuple of compressed), label

    def load(self, img):
        if not osp.exists(img):
            raise IOError("{img} does not exist in path".format(img=img))
        img_load = Image.open(img).convert("RGB")
        return img_load


class MultiClassificationGenerator(ImageGenerator):
    def __init__(self, logger = None, gpus=1, **kwargs):
        """Data generator for training and testing. Works with the <>. Should work with any crawler working on VeRi-like data. Not yet tested with VehicleID. Only  use with VeRi.

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
        super().__init__(logger, gpus=gpus, **kwargs)

    def buildDataset(
        self, datacrawler, mode: str, transform: List[object], **kwargs
    ) -> TorchDataset:
        if mode == "train":
            return MultiClassificationDataset(
                datacrawler.metadata[mode]["crawl"], transform, **kwargs
            )
        elif mode == "test":
            # For testing, we combine images in the query and testing set to generate batches
            return MultiClassificationDataset(
                datacrawler.metadata["val"]["crawl"]
                + datacrawler.metadata[mode]["crawl"],
                transform,
                **kwargs
            )
        elif mode == "full":
            return MultiClassificationDataset(
                datacrawler.metadata["val"]["crawl"]
                + datacrawler.metadata["train"]["crawl"]
                + datacrawler.metadata["test"]["crawl"],
                transform,
                **kwargs
            )
        else:
            raise NotImplementedError()

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        if mode == "train":
            return TorchDataLoader(
                dataset,
                batch_size=batch_size * self.gpus,
                shuffle=kwargs.get("shuffle", True),
                num_workers=self.workers,
                collate_fn=self.collate_simple,
            )
        elif mode == "test":
            return TorchDataLoader(
                dataset,
                batch_size=batch_size * self.gpus,
                shuffle=kwargs.get("shuffle", True),
                num_workers=self.workers,
                collate_fn=self.collate_simple,
            )
        elif mode == "full":
            return TorchDataLoader(
                dataset,
                batch_size=batch_size * self.gpus,
                shuffle=kwargs.get("shuffle", True),
                num_workers=self.workers,
                collate_fn=self.collate_simple,
            )
        else:
            raise NotImplementedError()

    def getNumEntities(self, datacrawler, mode, **kwargs):
        if mode in ["train", "test", "full"]:
            label_dict = {
                item: {"classes": datacrawler.metadata[mode]["classes"][item]}
                for item in kwargs.get("classificationclass", ["color"])
            }
            return LabelMetadata(label_dict=label_dict)
        else:
            raise NotImplementedError()

    def collate_simple(self, batch):
        img, class_ids, = zip(
            *batch
        )  # NOTE for deploy, this is a tuple, label, i.e. (img, img, img), class
        class_ids = torch.tensor(class_ids, dtype=torch.int64)
        return (
            torch.stack(img, dim=0),
            class_ids,
        )  # NOTE so this has to be modified...for tuple members...
