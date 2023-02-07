from glob import glob
import random, warnings
from typing import Any, List, Tuple
import torch, os, shutil
from torch.utils.data import TensorDataset, IterableDataset, Dataset
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None
    warnings.warn(
        "h5py not installed. This may cause issues with caching in HFGenerator."
    )
from tqdm import tqdm


class HFDataset(Dataset):
    def __init__(
        self, logger, dataset: List[Tuple[Any]], mode: str, transform=None, **kwargs
    ):
        """Initializes the HFDataset object.

        Args:
             logger (_type_): The logger object
             dataset (_type_): The crawler's metadata object
             mode (_type_): Mode to extract crawler contents
             transform (_type_, optional): Unused. Defaults to None.

         Kwargs (DO NOT CHANGE IN CONFIG):
             tokenizer: The tokenizer object

         COnfig Args (provided through DATASET_KWARGS):
             data_shuffle: Whether to shuffle the samples.

             cache: Cache preprocessed samples in an h5 file. Samples are accessed ad-hoc.
             memcache: Cache preprocessed samples in-memory. Samples are accessed ad-hoc.
             shardcache: Cache preprocessed samples in shards on disk. Samples are limited to
                 shard-based access.

             cache_replace: Whether to replace the h5 cache, if it exists.
             cachepath: Constructor for the directory where the cache is kept. This
                 is appended with the `mode`. So, for training, a `cachepath` of
                 "datashard-artifacts" becomes "datashard-artifacts-train"
             cachename: The name for the cache file stored inside the `cachepath`.
                 So a cachepath of name "hf-cache" means the following file is created:
                     "datashard-artifacts-train/hf-cache.h5"


             shardsize: Number of samples in each shard. This in turn determines
                 how much RAM is dedicated to hosting a shard in-memory when it is loaded.
             shard_replace: If shards already exist, whether to replace them.
             shardpath: Constructor for the directory where shards are kept. This
                 is appended with the `mode`. So, for training, a `shardpath` of
                 "datashard-artifacts" becomes "datashard-artifacts-train"
             shard-name: The name for each shard stored inside the `shardpath`.
                 This value is appended with the shard number. So a training shard
                 with index 5 and name "hf-shard" is stored at:
                     "datashard-artifacts-train/hf-shard5.pt"

             maxlen: The maximum length of the text to encode. Usually 512.
             mlm_probability: The probability of masking if using word and token masking.
                 Usually around 0.15
             masking (bool): Whether to use any masking. Specific masking options are next. Can also be `train_only` for masking during training only.
             token_mask: Whether to use random token masking, with `mlm_probability`
             word_mask: Whether to use random word masking, with `mlm_probability`
             keytoken_mask: Whether to use specific token masking, with `keytokens`. Can be a tuple with proxy lookup.
             keyword_mask: Whether to use specific word masking, with `keywords`. Can be a tuple with proxy lookup.
             keywords: What words to mask. Can be a tuple with proxy lookup.
             keytokens: What tokens to mask. Can be a tuple with proxy lookup.
             stopword_mask: Whether to use a stopword mask.
             label_idxs: List of indices for labels. Defaults to [1].
             annotation_idxs: List of indices for inputs. Defaults to [].



        """
        self.dataset = dataset  # list of tuples or dict (text, *text, *labels)
        self.logger = logger
        self.file_len = len(dataset)
        self.data_shuffle = kwargs.get("data_shuffle", True)

        # Cacheing options
        self.cache = kwargs.get("cache", False)
        self.memcache = kwargs.get("memcache", False)
        self.shardcache = kwargs.get("shardcache", False)

        if int(self.cache) + int(self.memcache) + int(self.shardcache) > 1:
            raise RuntimeError("Use only one cache type")

        self.word_masking = kwargs.get("word_mask", False)
        self.keyword_masking = kwargs.get("keyword_mask", True)
        self.token_masking = kwargs.get("token_mask", False)
        self.keytoken_masking = kwargs.get("keytoken_mask", False)
        self.stopword_masking = kwargs.get("stopword_mask", False)

        self.label_idxs = kwargs.get("label_idxs", [1])
        self.annotation_idxs = kwargs.get("annotation_idxs", self.label_idxs)

        # Tokenizer and lengths
        self.tokenizer = kwargs.get("tokenizer")
        self.maxlen = kwargs.get("maxlen")
        self.mlm = kwargs.get("mlm_probability", 0.2)
        self.masking = kwargs.get("masking", mode == "train")
        if self.masking == "train_only":
            self.masking = mode == "train"

        self.keywords = []
        if self.keyword_masking:
            self.keywords = kwargs.get("keywords", [])
        self.keytokens = []
        if self.keytoken_masking:
            self.keytokens = kwargs.get("keytokens", [])
        # Store indices corresponding to keytokens
        vocab = self.tokenizer.get_vocab()
        self.keytokens = {vocab[item]: 1 for item in self.keytokens if item in vocab}
        self.keytokens_tensor = torch.tensor(list(self.keytokens.keys())).unsqueeze(0).T

        # Cache options
        self.cache_replace = kwargs.get("cache_replace", False)
        self.cachepath = kwargs.get("cachepath", "artifacts")
        self.cachepath = mode + "-" + self.cachepath
        self.cachename = kwargs.get("cachename", "h5-cache") + "-"  # the dash
        if self.cache:
            self.logger.info(
                "[Mode `{mode}`] Will look in path [{path}] for shards `{shards}[e].pt`".format(
                    path=self.cachepath, shards=self.cachename, mode=mode
                )
            )
            self.base_cachepath = os.path.join(self.cachepath, self.cachename) + ".h5"
            self.cache_exist = False
            if os.path.exists(self.base_cachepath):
                if self.cache_replace:
                    self.logger.info("Deleting existing cache")
                    os.remove(self.cachepath)
                else:
                    self.cache_exist = True
                    self.logger.info(
                        "Cache already exists and `cache_replace` is False"
                    )
            else:
                self.logger.info("Cache does not exist and will be created.")
                self.logger.info("Creating cachepath %s" % self.cachepath)
                os.makedirs(self.cachepath, exist_ok=True)
                with h5py.File(self.base_cachepath, "w") as hfile:
                    hfile.create_dataset(
                        name="all_input_ids", shape=(len(self.dataset), self.maxlen)
                    )

        # Shardcache options
        self.shardsize = kwargs.get("shardsize", 0)
        self.shardsaveindex = 0
        self.shard_replace = kwargs.get("shard_replace", False)
        self.shardpath = kwargs.get("shardpath", "datashard-artifacts")
        self.shardpath = mode + "-" + self.shardpath
        self.shardname = (
            kwargs.get("shardname", "fnc-filtermask-shard") + "-"
        )  # the dash
        if self.shardcache:
            self.logger.info(
                "[Mode `{mode}`] Will look in path [{path}] for shards `{shards}[e].pt`".format(
                    path=self.shardpath, shards=self.shardname, mode=mode
                )
            )
        self.base_shardpath = os.path.join(self.shardpath, self.shardname)
        self.shards_exist = False
        if os.path.exists(self.base_shardpath + "0.pt"):
            if self.shard_replace:
                self.logger.info("Deleting existing shards")
                shutil.rmtree(self.shardpath)
            else:
                self.shards_exist = True
                self.logger.info("Shards already exist and `shard_replace` is False")
        else:
            self.logger.info("Shards do not exist and will be created.")
        if self.shardcache:
            self.logger.info("Creating shardpath %s" % self.shardpath)
            os.makedirs(self.shardpath, exist_ok=True)

        # This is for memcache and shardcache
        self.getcount = 0
        self.shardcount = 0
        if self.memcache or self.cache:
            # Checks when we have processed entire dataset, refreshes mask ids when needed
            self.refresh_flag = len(dataset)
        if self.shardcache:
            # We need to refresh mask ids each time we load a shard.
            # This is ONLY for randomized masking, i.e. with MLM probability.
            self.refresh_flag = self.shardsize
            # this shuffles the internal shard index, so that we get a random example, instead of in order
            self.shard_internal_shuffle = []
            # This shuffles the shard index, so that each epoch, we load shards in different orders. We will set this up after setting up shards...
            self.shard_shuffle = []

        self.getter = (
            self.uncachedget
        )  # for each entry, directly convert to features and return
        if self.cache:
            self.getter = self.diskget
        elif self.memcache:
            self.getter = self.memget
        elif self.shardcache:
            self.getter = self.shardget

        if self.cache or self.memcache:
            # The actual cache-ing
            self.logger.info("Started mem caching")
            self.input_length_cache = []
            self.logger.info("Converting to features")
            # For memcache, save dataset inside self.cached_dataset.
            # Same for cache. Difference being, cached_dataset is either in memory or on disk!
            self.convert_to_features(
                self.dataset, self.tokenizer, maxlen=self.maxlen, cache=True
            )
            self.logger.info("Masking")
            self.memcached_dataset = self.refresh_mask_ids(cache=True)

        if self.shardcache:
            self.input_length_cache = []
            if not self.shards_exist:
                self.logger.info("Generating shards")
                self.shardsaveindex = self.sharded_convert_to_features(
                    self.dataset, self.tokenizer, maxlen=self.maxlen
                )  # save shards and get numshards
            else:
                self.shardsaveindex = len(glob(self.base_shardpath + "*.pt"))
            if self.shardsaveindex < 1:
                raise ValueError(
                    "`shardsaveindex` is {val}, which is less than permissible minimum value of 1".format(
                        val=self.shardsaveindex
                    )
                )
            self.logger.info("Obtained %i shards" % self.shardsaveindex)
            self.shard_load_index = (
                0  # self.shardsaveindex is the maximum number of shards
            )
            self.shard_shuffle = list(
                range(self.shardsaveindex)
            )  # count started from 0
            if self.data_shuffle:
                self.logger.info("Shuffling shard load order")
                random.shuffle(self.shard_shuffle)
            self.sharded_dataset = self.load_shard(
                self.shard_shuffle[self.shard_load_index]
            )
            self.current_shardsize = len(self.sharded_dataset)
            if self.masking:
                self.logger.info("Refreshing token masks for loaded shard")  # TODO
                self.sharded_dataset = self.sharded_refresh_mask_ids(
                    self.sharded_dataset
                )  # TODO implement masking (and input_length_cache) for sharding
            self.shard_internal_shuffle = list(
                range(self.current_shardsize)
            )  # count started from 0
            if self.data_shuffle:
                random.shuffle(self.shard_internal_shuffle)

    def shardget(self, idx):
        """Given an index, retrieve a single sample from the post-processed
        dataset.

        We retrieve from shards, so we need to track the current shard we are
        working with, as well as which sample in the shard we are retrieving.

        THus, given an index, we do not use that index directly. Instead,
        HFGenerator tracks samples itself by using `self.shard_shuffle`
        and `self.internal_shard_shuffle`. The former tracks which shards
        have been accessed already, and the latter tracks which samples
        in the current shard have been tracked already.

        `self.getcount` tracks the current index.

        Shuffling if controlled with the `data_shuffle` argument during
        initialization of the HFDataset.

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # get entry from already loaded shardcache. so idx is incremented one by one -- No shuffling in data loader.
        # This is a design choice, and we can't really do anything if user does shuffle + sharding
        self.last_idx = idx
        response = self.sharded_dataset[self.shard_internal_shuffle[self.getcount]]
        self.getcount += 1

        if (
            self.getcount == self.current_shardsize
        ):  # we have exhausted examples in this shard
            self.getcount = 0
            self.shard_load_index += 1  # increment the shard index that we will load
            if (
                self.shard_load_index == self.shardsaveindex
            ):  # we have processed all shards.
                self.shard_load_index = 0
                if self.data_shuffle:
                    random.shuffle(self.shard_shuffle)
            self.sharded_dataset = self.load_shard(
                self.shard_shuffle[self.shard_load_index]
            )

            self.current_shardsize = len(self.sharded_dataset)
            self.shard_internal_shuffle = list(
                range(self.current_shardsize)
            )  # count started from 0
            if self.data_shuffle:
                random.shuffle(self.shard_internal_shuffle)
            if self.masking:
                self.sharded_dataset = self.sharded_refresh_mask_ids(
                    self.sharded_dataset
                )  # TODO implement masking (and input_length_cache) for sharding

        return response

    def sharded_convert_to_features(self, dataset, tokenizer, maxlen):
        features = []
        shardsaveindex = 0
        input_length_cache = []
        input_word_length_cache = []
        self.logger.info("Generating shards")
        pbar = tqdm(total=int(self.file_len / self.shardsize) + 1)
        # TODO
        # We basically need to deal with having multiple inputs and multiple labels...
        # Ok, so, we already know the structure of the tuple that dataset gets
        # Then, all we need is in config, specify additional_input_idx --> list | None
        # if None, we don't need to do anything
        # if a list, then our input is actually a tuple
        # So, if a list, what do we do...?
        #
        #
        #
        #
        #
        for idx, sample in enumerate(
            dataset
        ):  # This is a tuple with (text, ..., stuff)
            # Identify the indices of specific keywords to mask
            # if self.keyword_masking or self.word_masking:
            # NOTE: we assume that text is at index 0!!!!!
            word_tokens = sample[0].lower().split(" ")
            encoded_word_length = len(word_tokens)

            if self.masking and self.keyword_masking:
                keyword_idx = []
                for widx, item in enumerate(word_tokens):
                    if sum([1 if kword in item else 0 for kword in self.keywords]) > 0:
                        keyword_idx.append(
                            widx
                        )  # i.e. find word index for each keyword, if it exists

            encoded = self.tokenizer(
                sample[0],
                return_tensors="pt",
                padding="max_length",
                max_length=maxlen,
                truncation=True,
                return_length=True,
            )
            encoded_token_length = torch.sum(encoded["attention_mask"])
            encoded_word_ids = encoded.word_ids(0)
            input_length_cache.append(encoded_token_length)
            input_word_length_cache.append(encoded_word_length)
            # create a list of lists. Each sublist is a mask for each keyword. So, we can AND all sublists to get the overall mask.
            # Edge cases -- no keywords:
            merged_mask = encoded["attention_mask"]
            if self.masking and self.keyword_masking:
                match_idxs = torch.LongTensor(
                    [
                        [word_idx != idx_of_keyword for word_idx in encoded_word_ids]
                        for idx_of_keyword in keyword_idx
                    ]
                )
                if (
                    match_idxs.shape[0] > 0
                ):  # i.e. we have a mask for keywords, so we will and everything
                    match_idxs = torch.all(match_idxs, dim=0, keepdim=True)
                    merged_mask = torch.where(match_idxs == 0, match_idxs, merged_mask)

            if self.masking and self.keytoken_masking:
                # Dict with relevant keyword idxs...self.keytokens
                # First set 1 for all tokens, and 0 for masking tokens
                match_idxs = encoded["input_ids"].repeat((len(self.keytokens), 1))
                match_idxs = torch.all(
                    match_idxs != self.keytokens_tensor, dim=0, keepdim=True
                )
                merged_mask = torch.where(match_idxs == 0, match_idxs, merged_mask)

            features.append(
                (
                    encoded["input_ids"],
                    merged_mask,
                    encoded["token_type_ids"],
                    encoded_token_length,
                    encoded_word_length,
                    encoded_word_ids,
                    [
                        sample[label_idx] for label_idx in self.label_idxs
                    ],  # list of labels
                    [sample[annot_idx] for annot_idx in self.annotation_idxs],
                )  # list of secondary inputs. Defaults to empty.
            )

            if (idx + 1) % self.shardsize == 0:  # we need to save this shard.
                # TODO need to handle multiple labels, later...
                all_input_ids = torch.cat([f[0] for f in features])
                # TODO
                # Secondary inputs are other numeric inputs
                # We check if secondary is provided by checking what is in  in features[7] (last element)
                # if None, then secondary inputs are not provided. We can make it a matrix of zeros
                # # if true, the secondary inputs are provided, in a list of numbers
                # Then we convert then to a nxd matrix (n = shardsize, d = number of secondary inputs)
                # all_secondary_inputs = None
                all_attention_mask = torch.cat([f[1] for f in features])
                all_token_type_ids = torch.cat([f[2] for f in features])
                all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
                all_word_lens = torch.tensor([f[4] for f in features], dtype=torch.long)
                all_word_ids = torch.tensor(
                    np.nan_to_num(
                        np.array([f[5] for f in features], dtype=float), nan=-1
                    ),
                    dtype=torch.long,
                )
                all_labels = torch.tensor(
                    [f[6] for f in features], dtype=torch.long
                )  # TODO -- implement multiple labels as a list, then squash here to one dim if single label.
                all_masklm = -1 * torch.ones(all_input_ids.shape, dtype=torch.long)
                if len(features[0][7]) > 0:
                    all_annotations = torch.tensor([f[7] for f in features])
                else:
                    all_annotations = torch.zeros((self.shardsize, 1))

                # Propagate input masking to output masking
                for midx in range(all_attention_mask.shape[0]):
                    masked_index = torch.where(
                        all_attention_mask[midx][: input_length_cache[midx]] == 0
                    )[0]
                    all_masklm[midx][masked_index] = all_input_ids[midx][
                        masked_index
                    ]  # Set the masking labels for these to the actual word index from all_input_ids
                input_length_cache = []
                input_word_length_cache = []

                # SAVE HERE
                self.save_shard(
                    shard=TensorDataset(
                        all_input_ids,
                        all_attention_mask,
                        all_token_type_ids,
                        all_masklm,
                        all_lens,
                        all_word_lens,
                        all_word_ids,
                        all_labels,
                        all_annotations,
                    ),
                    shard_index=shardsaveindex,
                )
                shardsaveindex += 1
                features = []
                pbar.update(1)
        # final shard...
        if len(features) > 0:
            all_input_ids = torch.cat([f[0] for f in features])
            all_attention_mask = torch.cat([f[1] for f in features])
            all_token_type_ids = torch.cat([f[2] for f in features])
            all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
            all_word_lens = torch.tensor([f[4] for f in features], dtype=torch.long)
            all_word_ids = torch.tensor(
                np.nan_to_num(np.array([f[5] for f in features], dtype=float), nan=-1),
                dtype=torch.long,
            )
            all_labels = torch.tensor([f[6] for f in features], dtype=torch.long)
            all_masklm = -1 * torch.ones(all_input_ids.shape, dtype=torch.long)
            if len(features[0][7]) > 0:
                all_annotations = torch.tensor([f[7] for f in features])
            else:
                all_annotations = torch.zeros((len(features), 1))

            # Propagate input masking to output masking
            for midx in range(all_attention_mask.shape[0]):
                masked_index = torch.where(
                    all_attention_mask[midx][: input_length_cache[midx]] == 0
                )[0]
                all_masklm[midx][masked_index] = all_input_ids[midx][
                    masked_index
                ]  # Set the masking labels for these to the actual word index from all_input_ids
            input_length_cache = []

            # SAVE HERE
            self.save_shard(
                shard=TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    all_token_type_ids,
                    all_masklm,
                    all_lens,
                    all_word_lens,
                    all_word_ids,
                    all_labels,
                    all_annotations,
                ),
                shard_index=shardsaveindex,
            )
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
        self.logger.debug("Loading shard %s" % shard_save_path)
        return torch.load(shard_save_path)

    def memget(self, idx):
        self.getcount += 1
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
                all_attention_mask[idx][
                    self.mask_ids[idx]
                ] = 0  # Set the masking words to 0, so we do not attend to it during prediction
                all_masklm[idx][self.mask_ids[idx]] = self.all_input_ids[idx][
                    self.mask_ids[idx]
                ]  # Set the masking labels for these to the actual word index from all_input_ids

            return TensorDataset(
                self.all_input_ids,
                all_attention_mask,
                self.all_token_type_ids,
                all_masklm,
                self.all_lens,
                self.all_labels,
            )
        else:
            return TensorDataset(
                self.all_input_ids,
                self.all_attention_mask,
                self.all_token_type_ids,
                self.all_masklm,
                self.all_lens,
                self.all_labels,
            )

    def build_mask_ids(self, input_length_cache):
        # for each element, we get a set of indices that are randomly selected...
        # Also, -2 and +1 take care of [cls] and [sep] not being masked
        return [
            (torch.randperm(inplength - 2) + 1)[: int(inplength * self.mlm)]
            for inplength in input_length_cache
        ]

    def build_whole_word_mask_ids(self, input_length_cache):
        # for each element, we get a set of indices that are randomly selected...
        # Also, -2 and +1 take care of [cls] and [sep] not being masked
        return [
            (torch.randperm(word_length) + 1)[: int(word_length * self.mlm)]
            for word_length in input_length_cache
        ]

        # So, given word_len, we can extract which ids need to be masked
        # But, now we need a way to, given the ID that needs to be masked, get back the original words.

    def sharded_refresh_mask_ids(self, sharded_dataset: TensorDataset):
        self.logger.debug("Refreshing mask ids")
        #                   0               1                   2                 3           4           5             6            7............8
        # TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_lens, all_word_lens, all_word_ids, all_labels, all_annotations)

        if self.token_masking or self.word_masking:
            # merged_masklm =  torch.stack([shard[3] for shard in sharded_dataset])
            all_masklm = torch.stack([shard[3] for shard in sharded_dataset])
            all_attention_mask = torch.stack([shard[1] for shard in sharded_dataset])
            if self.token_masking:
                self.logger.debug("Performing random token masking")
                mask_ids = self.build_mask_ids(
                    torch.stack([shard[4] for shard in sharded_dataset])
                )

                # all_masklm = torch.stack([shard[3] for shard in sharded_dataset])
                all_input_ids = torch.stack([shard[0] for shard in sharded_dataset])

                for idx in range(self.current_shardsize):
                    all_attention_mask[idx][
                        mask_ids[idx]
                    ] = 0  # Set the masking words to 0, so we do not attend to it during prediction
                    all_masklm[idx][mask_ids[idx]] = all_input_ids[idx][
                        mask_ids[idx]
                    ]  # Set the masking labels for these to the actual word index from all_input_ids
                # merged_masklm *= all_masklm
            if self.word_masking:
                self.logger.debug("Performing random word masking")
                # TODO
                masking_words = self.build_whole_word_mask_ids(
                    torch.stack([shard[5] for shard in sharded_dataset])
                )

                # all_masklm = torch.stack([shard[3] for shard in sharded_dataset])
                all_input_ids = torch.stack([shard[0] for shard in sharded_dataset])
                all_word_ids = torch.stack([shard[6] for shard in sharded_dataset])
                for idx in range(self.current_shardsize):

                    match_idxs = torch.LongTensor(
                        [
                            [
                                word_idx != idx_of_keyword
                                for word_idx in all_word_ids[idx]
                            ]
                            for idx_of_keyword in masking_words[idx]
                        ]
                    )
                    if (
                        match_idxs.shape[0] > 0
                    ):  # i.e. we have a mask for keywords, so we will and everything
                        match_idxs = torch.all(match_idxs, dim=0, keepdim=True)
                        merged_mask = torch.where(match_idxs[0] == 0)[0]

                        all_attention_mask[idx][
                            merged_mask
                        ] = 0  # Set the masking words to 0, so we do not attend during prediction
                        all_masklm[idx][merged_mask] = all_input_ids[idx][
                            merged_mask
                        ]  # Set the masking labels for these to the actual word index from all_input_ids

                # merged_masklm *= all_masklm
            return TensorDataset(
                all_input_ids,
                all_attention_mask,
                torch.stack([shard[2] for shard in sharded_dataset]),
                all_masklm,
                torch.stack([shard[4] for shard in sharded_dataset]),
                torch.stack([shard[5] for shard in sharded_dataset]),
                torch.stack([shard[6] for shard in sharded_dataset]),
                torch.stack([shard[7] for shard in sharded_dataset]),
                torch.stack([shard[8] for shard in sharded_dataset]),
            )
        else:
            return sharded_dataset

    def convert_to_features(self, dataset, tokenizer, maxlen, cache=False):
        """Preprocess the dataset into text features. If cache is True, save the preprocessed data into a local h5 file.
        Else, save in memory.

        Args:
            dataset (_type_): _description_
            tokenizer (_type_): _description_
            maxlen (_type_): _description_
            cache (bool, optional): _description_. Defaults to False.
        """
        features = []
        self.input_length_cache = []
        for idx, sample in enumerate(dataset):
            tokens = self.tokenizer.tokenize(sample[0])
            if len(tokens) > maxlen - 2:
                tokens = tokens[0 : (maxlen - 2)]
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
            attention_mask = [1] * len(input_ids)
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
        self.all_attention_mask = torch.tensor(
            [f[1] for f in features], dtype=torch.long
        )
        self.all_token_type_ids = torch.tensor(
            [f[2] for f in features], dtype=torch.long
        )
        self.all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
        self.all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
        self.all_masklm = -1 * torch.ones(self.all_input_ids.shape, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.getter(idx)

    def uncachedget(self, idx):
        return


from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.generators import TextGenerator


class HFGenerator(TextGenerator):
    """HFGenerator is a generic text generator for HuggingFace-tokenizers
    and designed to work with HuggingFace style transformers.

    It can use any tokenizer that follows HuggingFace API.

    Usage:
        TODO
    """

    def build_transforms(self, transform, mode, **kwargs):  # <-- generator kwargs:
        """Use the GENERATOR_KWARGS to identify the tokenizer
        class. For now, only built-in tokenizers are supported.

        Args:
            transform (_type_): Unused. It is necessary here as a dummy argument in
                constructor for the core ednaml.generator API.
            mode (_type_): Whether to use train or test mode. Useful when selecting
                the correct subset from crawler.
        """
        from ednaml.utils import locate_class

        print("Building Transforms")
        tokenizer = kwargs.get("tokenizer", "HFAutoTokenizer")
        self.tokenizer = locate_class(
            package="ednaml",
            subpackage="utils",
            classpackage=tokenizer,
            classfile="tokenizers",
        )
        self.tokenizer = self.tokenizer(
            **kwargs
        )  # vocab_file, do_lower_case, spm_model_file

    def buildDataset(self, crawler, mode, transform, **kwargs):  # <-- dataset args:
        print("Building Dataset")
        return HFDataset(
            self.logger,
            crawler.metadata[mode]["crawl"],
            mode,
            tokenizer=self.tokenizer,
            **kwargs
        )  # needs maxlen, memcache, mlm_probability

    def buildDataLoader(self, dataset, mode, batch_size, **kwargs):
        print("Building Dataloader")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size * self.gpus,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def getNumEntities(self, crawler, mode, **kwargs):  # <-- dataset args
        label_dict = {
            item: {"classes": crawler.metadata[mode]["classes"][item]}
            for item in kwargs.get("classificationclass", ["color"])
        }
        return LabelMetadata(label_dict=label_dict)

    def collate_fn(self, batch):
        (
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_masklm,
            all_lens,
            all_word_lens,
            all_word_ids,
            all_labels,
            all_annotations,
        ) = map(torch.stack, zip(*batch))
        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        all_masklm = all_masklm[:, :max_len]
        # Right now, we don't care that much about all_word_ids or all_word_lens

        return (
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_masklm,
            all_annotations,
            all_labels,
        )
