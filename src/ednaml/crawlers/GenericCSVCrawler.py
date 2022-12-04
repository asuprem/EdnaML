import logging, os, csv, random
from ednaml.crawlers import Crawler
from ednaml.utils.web import download
from collections import Counter

class GenericCSVCrawler(Crawler):
    def __init__(self, logger: logging.Logger, csv_url = "", local_file = "", 
            label_columns = [], label_callbacks = [], data_callbacks = [], data_columns = [], 
            splits = [0.8, 0.1], balanced_split = False, balanced_split_label_index = 0, random_seed = 51444,
            headers_exist = True, delimiter = ",", label_names = [], label_classes = [], split_shuffle = True, shuffle = True):
        self.logger = logger

        if len(label_callbacks) == len(label_columns) or len(label_callbacks) == 1:
            pass
        else:
            raise ValueError("Length of label callbacks should match label columns or 1")
        
        if len(data_callbacks) == len(data_columns) or len(data_callbacks) == 1:
            pass
        else:
            raise ValueError("Length of label callbacks should match label columns or 1")

        callback_dictionary = {
            "int": int, "float": float, "str": str, "noop": lambda x:x
        }
        label_callback_lambdas = []
        data_callback_lambdas = []
        if len(label_callbacks) == 1:
            label_callback_lambdas = [callback_dictionary[label_callbacks[0]]]*len(label_columns)
        else:
            label_callback_lambdas = [callback_dictionary[item] for item in label_callbacks]
        
        if len(data_callbacks) == 1:
            data_callback_lambdas = [callback_dictionary[data_callbacks[0]]]*len(data_columns)
        else:
            data_callback_lambdas = [callback_dictionary[item] for item in data_callbacks]

        if csv_url == "":
            raise ValueError("No file or URL provided.")

        csv_path = ""
        if local_file == "":
            self.logger.info("No local path provided. Assuming `csv_url` is a file path itself")
            csv_path = csv_url
        else:
            
            if os.path.exists(local_file):
                self.logger.info("Skipping download. File already exists")
            else:
                csv_path = local_file
                download(csv_path, csv_url)
        dataset = []
        with open(csv_path, "r") as ffile:
            fobj = csv.reader(ffile, delimiter=delimiter)
            if headers_exist:
                header = next(fobj)
            for row in fobj:
                dataset.append(
                    (*[ data_callback_lambdas(row[item]) for idx, item in enumerate(data_columns)], *[label_callback_lambdas(row[item]) for idx, item in enumerate(label_columns)])
                )
        self.classes = {}
        if len(label_classes) == 0 or balanced_split:
            num_datacolumns = len(data_columns)
            label_counts = [Counter() for _ in range(len(label_columns))]
            [label_counts[idx].update(item) for sublist in dataset for idx, item in enumerate(sublist[num_datacolumns:])]
            for idx, label_name in enumerate(label_names):
                self.classes[label_name] = len(label_counts[idx])
        else:
            for idx, label_name in enumerate(label_names):
                self.classes[label_name] = label_classes[idx]

        self.metadata = {}
        self.metadata["train"] = {}
        self.metadata["test"] = {}
        self.metadata["val"] = {}
        self.metadata["train"]["crawl"] = []
        self.metadata["test"]["crawl"] = []
        self.metadata["val"]["crawl"] = []


        # Now that we have the dataset, we will do the splitting here
        random.seed(random_seed)
        if balanced_split:
            class_balanced_splits = [None for _ in range(self.classes[label_names[balanced_split_label_index]])]
            class_values = list(label_counts[balanced_split_label_index].keys())
            for class_idx in range(len(class_balanced_splits)):
                class_balanced_splits[class_idx] = [item for item in dataset if item[balanced_split_label_index] == class_values[class_idx]]
                if split_shuffle:
                    random.shuffle(class_balanced_splits[class_idx])
                train_split = int(len(class_balanced_splits[class_idx]) * splits[0])
                val_split = int(len(class_balanced_splits[class_idx]) * splits[1])
                
                self.metadata["train"]["crawl"] += class_balanced_splits[class_idx][:train_split]
                self.metadata["test"]["crawl"] += class_balanced_splits[class_idx][train_split: train_split+val_split]
                self.metadata["val"]["crawl"] += class_balanced_splits[class_idx][train_split+val_split:]
        else:

            if split_shuffle:
                random.shuffle(dataset)
            
            train_split = int(len(dataset) * splits[0])
            val_split = int(len(dataset) * splits[1])


            self.metadata["train"]["crawl"] = dataset[:train_split]
            self.metadata["test"]["crawl"] = dataset[train_split: train_split+val_split]
            self.metadata["val"]["crawl"] = dataset[train_split+val_split:]

        if shuffle:
            random.shuffle(self.metadata["train"]["crawl"])
            random.shuffle(self.metadata["test"]["crawl"])
            random.shuffle(self.metadata["val"]["crawl"])







        

        
        

