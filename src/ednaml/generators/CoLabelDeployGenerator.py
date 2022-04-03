from io import BytesIO
import os.path as osp
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
import torchvision.transforms as T


class CoLabelDeployDataset(TorchDataset):
    def __init__(self, dataset, transform=None, jpeg=[]):
        self.dataset = dataset
        self.transform = transform
        self.jpeg = jpeg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            [self.transform(item) for item in self.load(self.dataset[idx][0])],
            self.dataset[idx][1],
        )  # NOTE make this return a tuple for deploy (tuple of compressed), label

    def load(self, img):
        if not osp.exists(img):
            raise IOError("{img} does not exist in path".format(img=img))
        img_load = Image.open(img)
        jpeg = [self.JPEGcompression(img_load, qf).convert("RGB") for qf in self.jpeg]
        return [img_load.convert("RGB")] + jpeg

    def JPEGcompression(self, image: Image.Image, qf=90) -> Image.Image:
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


class CoLabelDeployGenerator:
    def __init__(
        self,
        gpus,
        i_shape=(208, 208),
        normalization_mean=0.5,
        normalization_std=0.5,
        normalization_scale=1.0 / 255.0,
        h_flip=0.5,
        t_crop=True,
        rea=True,
        **kwargs
    ):
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
        transformer_primitive.append(T.ToTensor())
        transformer_primitive.append(
            T.Normalize(mean=normalization_mean, std=normalization_std)
        )
        self.transformer = T.Compose(transformer_primitive)

        self.jpegs = kwargs.get("JPEG")

    def setup(
        self,
        datacrawler,
        mode="train",
        batch_size=32,
        instance=8,
        workers=8,
        crawlerclass="color",
    ):
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
            raise NotImplementedError()
        elif mode == "test":
            # For testing, we combine images in the query and testing set to generate batches
            self.__dataset = CoLabelDeployDataset(
                datacrawler.metadata["val"]["crawl"]
                + datacrawler.metadata[mode]["crawl"],
                self.transformer,
                self.jpegs,
            )
        elif mode == "full":
            self.__dataset = CoLabelDeployDataset(
                datacrawler.metadata["val"]["crawl"]
                + datacrawler.metadata["train"]["crawl"]
                + datacrawler.metadata["test"]["crawl"],
                self.transformer,
                self.jpegs,
            )
        else:
            raise NotImplementedError()

        if mode == "train":
            raise NotImplementedError()
        elif mode == "test":
            self.dataloader = TorchDataLoader(
                self.__dataset,
                batch_size=batch_size * self.gpus,
                shuffle=True,
                num_workers=self.workers,
                collate_fn=self.collate_simple,
            )
            self.num_entities = datacrawler.metadata[mode]["classes"][crawlerclass]
        elif mode == "full":
            self.dataloader = TorchDataLoader(
                self.__dataset,
                batch_size=batch_size * self.gpus,
                shuffle=True,
                num_workers=self.workers,
                collate_fn=self.collate_simple,
            )
            self.num_entities = datacrawler.metadata[mode]["classes"][crawlerclass]
        else:
            raise NotImplementedError()

    def collate_simple(self, batch):
        img, label = zip(
            *batch
        )  # NOTE for deploy, this is a tuple, label, i.e. (img, img, img), class
        num_stacks = len(img[0])
        num_tstacks = [None] * num_stacks
        for idx in range(num_stacks):
            num_tstacks[idx] = torch.stack([item[idx] for item in img], dim=0)

        class_id = torch.tensor(label, dtype=torch.int64)
        return (
            tuple(num_tstacks),
            class_id,
        )  # NOTE so this has to be modified...for tuple members...
