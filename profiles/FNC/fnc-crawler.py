from typing import Any, Dict, List
import ednaml
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna
import json, os
from ednaml.crawlers import Crawler
from ednaml.utils.web import download
from ednaml.utils.file_utils import IterableFile
import gzip
import shutil


# This is to read from Azure...download a thing from azure, and unzip, and load it into memory...
# We will datashard it with the Datareader args, specifically for the Deployment...


@edna.register_crawler
class FNCCrawler(Crawler):
    """Downloads a FNC raw file from Azure if it exists and builds an iterable from the file contents.

    Crawler yields a list of tweet objects, i.e. a dictionary containing tweet attributes.
    """
    def __init__(
        self,
        logger=None,
        azstorage="ednadatasets",
        azcontainer="edna-covid-raw",
        azfile="tweets-2020-01-22.json.gz",
        test_file = None,
        train_azstorage=None,
        train_azcontainer=None,
        train_azfile=None,
        train_file = None
    ):
        """Initializes the FNC Crawler.

        By default, only the test data is populated. If the corresponding values for arguments is None, it is not used, instead of throwing error.

        Args:
            logger (_type_, optional): Logger. Defaults to None.

            azstorage (str, optional): Axure storage name. Defaults to "ednadatasets".
            azcontainer (str, optional): Azure container name. Defaults to "edna-covid-raw".
            azfile (str, optional): Azure file name (blob name). Defaults to "tweets-2020-01-22.json.gz".
            test_file (_type_, optional): If using local files, this will supersede the Azure data. Defaults to None.

            train_azstorage (str, optional): Axure storage name. Defaults to "ednadatasets".
            train_azcontainer (str, optional): Azure container name. Defaults to "edna-covid-raw".
            train_azfile (str, optional): Azure file name (blob name). Defaults to "tweets-2020-01-22.json.gz".
            train_file (_type_, optional): If using local files, this will be present in the "train" metadata and supersede Azure data. Defaults to None.
            
        """
        if test_file is None:
            if azstorage is not None:
                az_url = self.build_url(azstorage, azcontainer, azfile)
                logger.info("Crawling %s" % (az_url))
                if not os.path.exists(azfile):
                    download(azfile, az_url)
                else:
                    logger.info("%s already exists at %s" % (az_url, azfile))
                az_jsonfile = os.path.splitext(azfile)[0]

                # Then unzip the file
                if not os.path.exists(az_jsonfile):
                    with gzip.open(azfile, "rb") as f_in:
                        with open(az_jsonfile, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
            else:
                az_jsonfile = None
        else:   # TODO possible sanity checks????
            az_jsonfile = test_file


        if train_file is None:
            if train_azstorage is not None:
                train_az_url = self.build_url(train_azstorage, train_azcontainer, train_azfile)
                logger.info("Crawling %s" % (train_az_url))
                if not os.path.exists(train_azfile):
                    download(train_azfile, train_az_url)
                else:
                    logger.info("%s already exists at %s" % (train_az_url, train_azfile))
                train_az_jsonfile = os.path.splitext(train_azfile)[0]

                # Then unzip the file
                if not os.path.exists(train_az_jsonfile):
                    with gzip.open(train_azfile, "rb") as f_in:
                        with open(train_az_jsonfile, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
            else:
                train_az_jsonfile = None
        else:   # TODO possible sanity checks????
            train_az_jsonfile = train_file

        # set up class metadata
        self.classes = {}
        self.classes["fnews"] = 2

        # set up content metadata
        self.metadata = {}
        self.metadata["secondary"] = {}

        if az_jsonfile is not None:
            self.metadata["secondary"]["linecount"] = self.bufcount(az_jsonfile)
        else:
            self.metadata["secondary"]["linecount"] = 0
        if train_az_jsonfile is not None:
            self.metadata["secondary"]["train_linecount"] = self.bufcount(train_az_jsonfile)
        else:
            self.metadata["secondary"]["train_linecount"] = 0

        self.metadata["train"] = {}
        self.metadata["test"] = {}
        self.metadata["val"] = {}
        self.metadata["train"]["crawl"] = []
        self.metadata["test"]["crawl"] = []
        self.metadata["val"]["crawl"] = []

        # Get the tweet data as list, <id, text, url> TODO
        # basically need a generator that yields results and maps them through some function
        # This will be slow because of json.loads line by line. We should speed this up with chunking?

        # TODO extracts full_text only. Fix this so it extracts all relevant objects (or possibly all objects??)
        if az_jsonfile is not None:
            self.metadata["test"]["crawl"] = IterableFile(
                az_jsonfile,
                #line_callback=lambda row: (json.loads(row.strip())["full_text"], 0),
                line_callback=lambda row: json.loads(row.strip()),
            )
        if train_az_jsonfile is not None:
            self.metadata["train"]["crawl"] = IterableFile(
                train_az_jsonfile,
                #line_callback=lambda row: (json.loads(row.strip())["full_text"], 0),
                line_callback=lambda row: json.loads(row.strip()),
            )
        self.metadata["train"]["classes"] = self.classes
        self.metadata["test"]["classes"] = self.classes
        self.metadata["val"]["classes"] = self.classes
        # pdb.set_trace()

    def build_url(self, azstorage, azcontainer, azfile):
        return "https://{azstorage}.blob.core.windows.net/{azcontainer}/{azfile}".format(
            azstorage=azstorage, azcontainer=azcontainer, azfile=azfile
        )

    def bufcount(self, filename):
        f = open(filename)                  
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read # loop optimization

        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)

        return lines



import torch
from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.generators import TextGenerator
from torch.utils.data import Dataset as TorchDataset
import torch.utils.data
import logging
from glob import glob

class FNCRawDataset(TorchDataset):
    """FNCRawDataset converts the raw file into datashards.

    Even if there is no tokenizer performing any conversion, our files are large
    and may not fit in memory in many cases.

    So, FNCRawDataset will iterate through the Crawler's file iterable, save shards in individual files
    and record the overall length.

    Since there is no tokenization, there is also no shuffling, since we are not training.


    Args:
        TorchDataset (_type_): _description_
    """
    def __init__(self, logger: logging.Logger, crawler_list: IterableFile, mode = "test", **kwargs):
        self.dataset = crawler_list
        self.logger = logger

        self.shardsize = kwargs.get("shardsize", 5000)
        self.shardsaveindex = 0
        self.shard_replace = kwargs.get("shard_replace", False)


        self.shardpath = kwargs.get("shardpath", "datashard-artifacts")
        self.shardpath = mode+"-"+self.shardpath    #e.g. test-datashard-artifacts

        self.shardname = kwargs.get("shardname", "fnc-shard") + "-"  #the dash
        self.base_shardpath = os.path.join(self.shardpath, self.shardname)
        self.shards_exist = False

        if os.path.exists(self.base_shardpath + "0.pt"):
            if self.shard_replace:
                self.logger.debug("Deleting existing shards")
                shutil.rmtree(self.shardpath)
            else:
                with open(os.path.join(self.shardpath,"len.txt"), "r") as f:
                    self.len = int(f.read().strip())
                self.shards_exist = True
                self.logger.debug("Shards already exist and `shard_replace` is False")
        os.makedirs(self.shardpath, exist_ok=True)

        # to track what we have used __getitem__ for so far...
        self.getcount = 0
        # Not sure what to do with this yet...
        self.refresh_flag = self.shardsize

        if not self.shards_exist:
            self.shardsaveindex = self.sharded_convert_to_features(self.dataset)    # save shards and get numshards
        else:
            self.shardsaveindex = len(glob(os.path.join(self.shardpath, "*.pt")))-1   # Get the total number of shards.

        self.shard_load_index = 0   # self.shardsaveindex is the maximum number of shards
        self.shard_shuffle = list(range(self.shardsaveindex))    # count started from 0. We will not shuffle this
        #if self.data_shuffle:
        #    random.shuffle(self.shard_shuffle)
        self.sharded_dataset = self.load_shard(self.shard_shuffle[self.shard_load_index])
        self.current_shardsize = len(self.sharded_dataset)  # because load_shard returns a list...
        self.shard_internal_shuffle = list(range(self.current_shardsize))    #count started from 0; we will not shuffle this.
        


    def sharded_convert_to_features(self, dataset: IterableFile):
        """We convert the dataset, an IterableFile, into shards here. Each shard is basically a 
        json file containing a fixed number of elements.

        Args:
            dataset (IterableFile): _description_

        Returns:
            _type_: _description_
        """
        shardsaveindex = 0
        #pbar = tqdm(total=int(len(dataset)/self.shardsize)+1)
        trow = []
        for idx, row in enumerate(dataset):
            trow.append(row)

            if (idx+1)%self.shardsize == 0: # we need to save this shard.
                
                self.save_shard(shard=trow,
                        shard_index = shardsaveindex)
                shardsaveindex += 1
                trow = []
                #pbar.update(1)
        # final shard...
        
        if len(trow) > 0:
            # SAVE HERE
            self.save_shard(shard=trow,
                    shard_index = shardsaveindex)
            shardsaveindex += 1
            trow = []
            #pbar.update(1)
        #pbar.close()
        self.len = idx
        with open(os.path.join(self.shardpath,"len.txt"), "w") as f:
            f.write(str(self.len))
        return shardsaveindex

    def save_shard(self, shard: List[Dict[str,str]], shard_index: int):
        shard_save_path = self.base_shardpath + str(shard_index) + ".pt"
        with open(shard_save_path, 'w') as f:
            json.dump(shard, f, ensure_ascii=False)
        #json.dump(shard, shard_save_path)

    def load_shard(self, shard_index: int):
        shard_save_path = self.base_shardpath + str(shard_index) + ".pt"
        self.logger.debug("Loading shard %s"%shard_save_path)
        with open(shard_save_path, 'r') as f:
          jload = json.load(f)
        return jload




    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.getcount += 1
        if self.getcount == self.current_shardsize:   # we have exhausted examples in this shard
            self.getcount = 0
            self.shard_load_index += 1                  # increment the shard index that we will load
            if self.shard_load_index == self.shardsaveindex: # we have processed all shards.
                self.shard_load_index = 0
                #if self.data_shuffle:
                #    random.shuffle(self.shard_shuffle)

            self.sharded_dataset = self.load_shard(self.shard_shuffle[self.shard_load_index])   #There is no shuffle, so we are fine...
            self.current_shardsize = len(self.sharded_dataset)
            self.shard_internal_shuffle = list(range(self.current_shardsize))    #count started from 0
            #if self.data_shuffle:
            #    random.shuffle(self.shard_internal_shuffle)
        #shardindex = idx % self.current_shardsize
        #return self.sharded_dataset[self.shard_internal_shuffle[shardindex]]
        return self.sharded_dataset[self.shard_internal_shuffle[self.getcount]]

@edna.register_generator
class FNCRawGenerator(TextGenerator):
    """The FNCRawGenerator simply yields the raw text from FNCCrawler in batches. It does no processing through any encoder.

    Args:
        TextGenerator (_type_): _description_
    """

    def build_transforms(self, transforms: Dict[str, Any], mode, **kwargs):
        return None

    def buildDataset(self, datacrawler, mode: str, transform: List[object], **kwargs) -> TorchDataset:
        return FNCRawDataset(self.logger, datacrawler.metadata[mode]["crawl"], mode, **kwargs)

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        return torch.utils.data.DataLoader(
            dataset,batch_size=batch_size*self.gpus,
                    shuffle=(True if mode=="train" else kwargs.get("shuffle", False)), num_workers = self.workers,
                    collate_fn=self.collate_fn)
    
    def getNumEntities(self, crawler, mode, **kwargs):  #<-- dataset args
        label_dict = {
            item: {"classes": crawler.metadata[mode]["classes"][item]}
            for item in kwargs.get("classificationclass", ["fnews"])
        }
        return LabelMetadata(label_dict=label_dict)


    def collate_fn(self, batch):
        return batch
