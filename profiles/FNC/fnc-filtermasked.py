from glob import glob
import random
import torch, os, shutil
from torch.utils.data import TensorDataset
from tqdm import tqdm
import ednaml.core.decorators as edna




class FNCFilterMaskDataset(torch.utils.data.Dataset):
    def __init__(self, logger, dataset, mode, transform=None, **kwargs):
        self.dataset = dataset  # list of tuples (text, labels, labels)
        self.logger = logger
        self.file_len = len(dataset)    #kwargs.get("crawler_secondary").get("linecount", 1)
        self.data_shuffle = kwargs.get("data_shuffle", True)

        # Options
        self.cache = kwargs.get("cache", False)
        self.memcache = kwargs.get("memcache", False)

        self.shardcache = kwargs.get("shardcache", False)
        self.shardsize = kwargs.get("shardsize", 0)
        self.shardsaveindex = 0
        self.shard_replace = kwargs.get("shard_replace", False)
        self.shardpath = kwargs.get("shardpath", "datashard-artifacts")
        self.shardpath = mode+"-"+self.shardpath
        self.shardname = kwargs.get("shardname", "fnc-filtermask-shard") + "-"  #the dash
        if self.shardcache:
            self.logger.debug("[Mode `{mode}`] Will look in path [{path}] for shards `{shards}[e].pt`".format(path=self.shardpath, shards=self.shardname, mode=mode))
        self.base_shardpath = os.path.join(self.shardpath, self.shardname)
        self.shards_exist = False
        if os.path.exists(self.base_shardpath + "0.pt"):
            if self.shard_replace:
                self.logger.debug("Deleting existing shards")
                shutil.rmtree(self.shardpath)
            else:
                self.shards_exist = True
                self.logger.debug("Shards already exist and `shard_replace` is False")
        else:
          self.logger.debug("Shards do not exist and will be created.")
        if self.shardcache:
            self.logger.debug("Creating shardpath %s"%self.shardpath)
            os.makedirs(self.shardpath, exist_ok=True)
        
        
        
        self.word_masking = kwargs.get("word_mask", False)
        self.keyword_masking = kwargs.get("keyword_mask", True)
        self.token_masking = kwargs.get("token_mask", False)
        self.keytoken_masking = kwargs.get("keytoken_mask", False)
        
        
        # Tokenizer and lengths
        self.tokenizer = kwargs.get("tokenizer")
        self.maxlen = kwargs.get("maxlen")
        self.mlm = kwargs.get("mlm_probability", 0.2)
        self.masking = kwargs.get("masking", mode=="train")


        self.keywords = []
        if self.keyword_masking:
            self.keywords = kwargs.get("keywords", [])
        self.keytokens = []
        if self.keytoken_masking:
            self.keytokens = kwargs.get("keytokens", [])
        # Store indices corresponding to keytokens
        vocab = self.tokenizer.get_vocab()
        self.keytokens = {vocab[item]:1 for item in self.keytokens if item in vocab}

        # This is for memcache and shardcache
        self.getcount = 0
        self.shardcount = 0
        if self.memcache:
            self.refresh_flag = len(dataset)    # Checks when we have processed entire dataset, refreshes mask ids when needed
        if self.shardcache:
            self.refresh_flag = self.shardsize
            # this shuffles the internal shard index, so that we get a random example, instead of in order
            self.shard_internal_shuffle = []
            # This shuffles the shard index, so that each epoch, we load shards in different orders. We will set this up after setting up shards...
            self.shard_shuffle = []

        if int(self.cache) + int(self.memcache) + int(self.shardcache) > 1:
            raise ValueError("Use only one cache type")
        
        self.getter = self.uncachedget  # for each entry, directly convert to features and return
        if self.cache:
            raise NotImplementedError()
            self.getter = self.diskget
        elif self.memcache:
            self.getter = self.memget
        elif self.shardcache:
            self.getter = self.shardget

        # if cache, then we will cache transformations to disk, then load them
        # if memcache, then we will cache to memory and load them
        # if shardcache, we will cache to disk in a sharded fashion, and load shards

        if self.cache or self.memcache:
        # The actual cache-ing
            self.logger.debug("Started mem caching")
            self.input_length_cache = []
            self.logger.debug("Converting to features")
            self.convert_to_features(self.dataset, self.tokenizer, maxlen=self.maxlen)
            self.logger.debug("Masking")
            self.memcached_dataset = self.refresh_mask_ids()

        if self.shardcache:
            self.input_length_cache = []
            if not self.shards_exist:
                self.logger.debug("Generating shards")
                self.shardsaveindex = self.sharded_convert_to_features(self.dataset, self.tokenizer, maxlen=self.maxlen)    # save shards and get numshards
            else:
                self.shardsaveindex = len(glob(os.path.join(self.shardpath, "*.pt")))-1   # TODO Bug fix if files do not have consistent numbering
            self.logger.debug("Obtained %i shards"%self.shardsaveindex)
            self.shard_load_index = 0   # self.shardsaveindex is the maximum number of shards
            self.shard_shuffle = list(range(self.shardsaveindex))    # count started from 0
            if self.data_shuffle:
                self.logger.debug("Shuffling shard load order")
                random.shuffle(self.shard_shuffle)
            self.sharded_dataset = self.load_shard(self.shard_shuffle[self.shard_load_index])
            if self.masking:
                self.logger.debug("Refreshing token masks for loaded shard")    # TODO
                self.sharded_dataset = self.refresh_mask_ids(self.sharded_dataset)  # TODO implement masking (and input_length_cache) for sharding
            self.current_shardsize = len(self.sharded_dataset)
            self.shard_internal_shuffle = list(range(self.current_shardsize))    #count started from 0
            if self.data_shuffle:
                random.shuffle(self.shard_internal_shuffle)



    def shardget(self, idx):
        # get entry from already loaded shardcache. so idx is incremented one by one -- No shuffling in data loader.
        # This is a design choice, and we can't really do anything if user does shuffle + sharding
        self.getcount += 1
        if self.getcount == self.current_shardsize:   # we have exhausted examples in this shard
            self.getcount = 0
            self.shard_load_index += 1                  # increment the shard index that we will load
            if self.shard_load_index == self.shardsaveindex: # we have processed all shards.
                self.shard_load_index = 0
                if self.data_shuffle:
                    random.shuffle(self.shard_shuffle)
            self.sharded_dataset = self.load_shard(self.shard_shuffle[self.shard_load_index])

            if self.masking:
                self.sharded_dataset = self.refresh_mask_ids(self.sharded_dataset)  # TODO implement masking (and input_length_cache) for sharding

            self.current_shardsize = len(self.sharded_dataset)
            self.shard_internal_shuffle = list(range(self.current_shardsize))    #count started from 0
            if self.data_shuffle:
                random.shuffle(self.shard_internal_shuffle)
        #shardindex = idx % self.current_shardsize
        return self.sharded_dataset[self.shard_internal_shuffle[self.getcount]]

    def sharded_convert_to_features(self, dataset, tokenizer, maxlen):
        features = []
        shardsaveindex = 0
        self.input_length_cache = []
        self.logger.info("Generating shards")
        pbar = tqdm(total=int(self.file_len/self.shardsize)+1)
        for idx, sample in enumerate(dataset):
            if self.keyword_masking:
                word_tokens = sample["full_text"].split(" ")
                keyword_idx = []
                for widx,item in enumerate(word_tokens):
                    if sum([1 if kword in item else 0 for kword in self.keywords]) > 0:
                        keyword_idx.append(widx) # i.e. find word index for each keyword, if it exists

            encoded = self.tokenizer(sample["full_text"], return_tensors="pt", padding="max_length", max_length = maxlen, truncation = True, return_length = True)
            enc_len = torch.sum(encoded["attention_mask"])
            self.input_length_cache.append(enc_len)
            # create a list of lists. Each sublist is a mask for each keyword. So, we can AND all sublists to get the overall mask.
            # Edge cases -- no keywords: 
            if self.keyword_masking:
                match_idxs = torch.LongTensor([[wid != keyword_idx[kidx] for wid in encoded.word_ids(0)] for kidx in range(len(keyword_idx))])
                if match_idxs.shape[0] > 0: #i.e. we have a mask for keywords, so we will and everything
                    match_idxs = torch.all(match_idxs, dim=0, keepdim=True)
                    merged_mask = torch.where(match_idxs == 0, match_idxs, encoded["attention_mask"])
                else:
                    merged_mask = encoded["attention_mask"]
            else:
                merged_mask = encoded["attention_mask"]
            #if self.keytoken_masking:

            
            features.append(
                (encoded["input_ids"], merged_mask, encoded["token_type_ids"], enc_len, 0)  # 0 used to be *sample[1:], i.e. the label
            )

            if (idx+1)%self.shardsize == 0: # we need to save this shard.
                # TODO need to handle multiple labels, later...
                all_input_ids = torch.cat([f[0] for f in features])
                all_attention_mask = torch.cat([f[1] for f in features])
                all_token_type_ids = torch.cat([f[2] for f in features])
                all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
                all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
                all_masklm = -1*torch.ones(all_input_ids.shape, dtype=torch.long)

                # Propagate input masking to output masking
                for midx in range(all_attention_mask.shape[0]):
                    masked_index = torch.where(all_attention_mask[midx][:self.input_length_cache[midx]]==0)[0]
                    all_masklm[midx][masked_index] = all_input_ids[midx][masked_index] # Set the masking labels for these to the actual word index from all_input_ids
                self.input_length_cache = []

                # SAVE HERE
                self.save_shard(shard=TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_lens, all_labels),
                        shard_index = shardsaveindex)
                shardsaveindex += 1
                features = []
                pbar.update(1)
        # final shard...
        if len(features) > 0:
            all_input_ids = torch.cat([f[0] for f in features])
            all_attention_mask = torch.cat([f[1] for f in features])
            all_token_type_ids = torch.cat([f[2] for f in features])
            all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
            all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
            all_masklm = -1*torch.ones(all_input_ids.shape, dtype=torch.long)

            # Propagate input masking to output masking
            for midx in range(all_attention_mask.shape[0]):
                masked_index = torch.where(all_attention_mask[midx][:self.input_length_cache[midx]]==0)[0]
                all_masklm[midx][masked_index] = all_input_ids[midx][masked_index] # Set the masking labels for these to the actual word index from all_input_ids
            self.input_length_cache = []

            # SAVE HERE
            self.save_shard(shard=TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_lens, all_labels),
                    shard_index = shardsaveindex)
            shardsaveindex += 1
            features = []
            pbar.update(1)
        pbar.close()
        return shardsaveindex

    def save_shard(self, shard: TensorDataset, shard_index: int):
        shard_save_path = self.base_shardpath + str(shard_index) + ".pt"
        torch.save(shard, shard_save_path)

    def load_shard(self, shard_index: int):
        shard_save_path = self.base_shardpath + str(shard_index) + ".pt"
        self.logger.debug("Loading shard %s"%shard_save_path)
        return torch.load(shard_save_path)

    def memget(self, idx):
        self.getcount+=1
        if self.getcount > self.refresh_flag and self.masking:
            self.memcached_dataset = self.refresh_mask_ids()
            self.getcount = 0
        return self.memcached_dataset[idx]

    def refresh_mask_ids(self):
        self.logger.debug("Refreshing mask ids")
        if self.masking:
            self.mask_ids = self.build_mask_ids(self.input_length_cache)
            
            all_attention_mask = self.all_attention_mask.clone()
            all_masklm = self.all_masklm.clone()

            for idx in range(self.refresh_flag):
                all_attention_mask[idx][self.mask_ids[idx]] = 0 # Set the masking words to 0, so we do not attend to it during prediction
                all_masklm[idx][self.mask_ids[idx]] = self.all_input_ids[idx][self.mask_ids[idx]] # Set the masking labels for these to the actual word index from all_input_ids

            return TensorDataset(self.all_input_ids, all_attention_mask, self.all_token_type_ids, all_masklm, self.all_lens, self.all_labels)
        else:
            return TensorDataset(self.all_input_ids, self.all_attention_mask, self.all_token_type_ids, self.all_masklm, self.all_lens, self.all_labels)

    def build_mask_ids(self, input_length_cache):
        # for each element, we get a set of indices that are randomly selected...
        # Also, -2 and +1 take care of [cls] and [sep] not being masked
        return [(torch.randperm(inplength-2)+1)[:int(inplength*self.mlm)]  for inplength in input_length_cache]

    
    def sharded_refresh_mask_ids(self):
        pass

    def sharded_build_mask_ids(self, input_length_cache):
        pass


    def convert_to_features(self, dataset, tokenizer, maxlen):
        features = []
        self.input_length_cache = []
        for idx, sample in enumerate(dataset):
            tokens = self.tokenizer.tokenize(sample[0])
            if len(tokens) > maxlen - 2:
                tokens = tokens[0:(maxlen - 2)]
            # An easy way potentially -- get the index of each word in the token set...
            # Then check which word is in keyword
            # Then mask out that word...????????
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
                (input_ids, attention_mask, token_type_ids, input_len, *sample[1:])
            )

        self.all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
        self.all_token_type_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
        self.all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
        self.all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
        self.all_masklm = -1*torch.ones(self.all_input_ids.shape, dtype=torch.long)
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.getter(idx)

    def uncachedget(self, idx):
        return 


from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.generators import TextGenerator

@edna.register_generator
class FNCFilterMaskGenerator(TextGenerator):
  # input includes tokenizer for build...

  # Set it up such that, given a crawler, create a dataset from it.
  # Then create a cached batch
  # From cached batch, yield batches until it is empty

    def build_transforms(self, transform, mode, **kwargs):  #<-- generator kwargs:
        from ednaml.utils import locate_class
        self.logger.info("Building Transforms")
        tokenizer = kwargs.get("tokenizer", "HFAutoTokenizer")
        self.tokenizer = locate_class(package="ednaml", subpackage="utils", classpackage=tokenizer, classfile="tokenizers")
        self.tokenizer = self.tokenizer(**kwargs) 

  
    def buildDataset(self, crawler, mode, transform, **kwargs): #<-- dataset args:
        self.logger.info("Building Dataset")
        return FNCFilterMaskDataset(self.logger, crawler.metadata[mode]["crawl"], mode, tokenizer = self.tokenizer, crawler_secondary = crawler.metadata.get("secondary", {}), **kwargs) # needs maxlen, memcache, mlm_probability

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        self.logger.info("Building Dataloader")
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size*self.gpus,
                                        shuffle=kwargs.get("shuffle", True), num_workers = self.workers, 
                                       collate_fn=self.collate_fn)

    def getNumEntities(self, crawler, mode, **kwargs):  #<-- dataset args
        label_dict = {
            item: {"classes": crawler.metadata[mode]["classes"][item]}
            for item in kwargs.get("classificationclass", ["color"])
        }
        return LabelMetadata(label_dict=label_dict)

    def collate_fn(self, batch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_lens, all_labels  = map(torch.stack, zip(*batch))
        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        all_masklm = all_masklm[:, :max_len]

        return all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels