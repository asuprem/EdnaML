#!/usr/bin/env python3
"""
This is the crawler for the CompCars dataset.

It stores tuples containing the CompCars annotations for each sample.

Tuple is of the form:

(make, model, releasedyear, type, path/to/image)

- makeid: starts at 0 in tuple. Starts at 1 in dataset. See classes["make"] for human readable names
- modelid: starts at 0 in tuple. Starts at 1 in dataset. See classes["model"] for human readable names
- type: starts at 0 in tuple. Starts at 1 in dataset. See classes["type"] for human readable names
- releasedyear: No change for now...

"""
import os
import re
import glob
from typing import Dict
from scipy.io import loadmat


class CoLabelCompCarsCrawler:
    """Data crawler for CompCars Data dataset (NOT for sv-data)
    """

    makeidx = 0
    modelidx = 1
    yearidx = 2
    typeidx = 3
    pathidx = 4

    def __init__(self, data_folder="CompCars", train_folder="image", **kwargs):
        self.metadata = {}
        self.classes = {}
        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)
        self.train_file = kwargs.get("trainfile")
        self.test_file = kwargs.get("testfile")

        self.logger = kwargs.get("logger")

        self.__verify(self.data_folder)
        self.__verify(self.train_folder)

        self.crawl()

    def __verify(self, folder):
        if not os.path.exists(folder):
            raise IOError(
                "Folder {data_folder} does not exist".format(data_folder=folder)
            )
        else:
            self.logger.info("Found {data_folder}".format(data_folder=folder))

    def crawl(self,):
        # self.classes = {}   #self.__getclasses(self.train_folder) # This is a map from human-readable class to numbers (i.e. the ids in the dataset...)

        # for each label we have, we label the self.matadata["train'}["classes"]["label"] = num-entities...

        # Now extract the hum,an-reasable labels...using the mat files
        human_readable = os.path.join(self.data_folder, "misc")
        makemodelmat = os.path.join(human_readable, "make_model_name.mat")
        typemat = os.path.join(human_readable, "car_type.mat")

        # Set up type humanr readable
        self.classes["type"] = {}
        self.classes["make"] = {}
        self.classes["model"] = {}
        tmat = loadmat(typemat)
        for idx in range(tmat["types"].shape[1]):  # --> 12
            self.classes["type"][idx] = tmat["types"][0, idx][0]
        mmmat = loadmat(makemodelmat)
        for idx in range(mmmat["model_names"].shape[0]):  # --> 12
            if mmmat["model_names"][idx, 0].shape[0] > 0:
                self.classes["model"][idx] = mmmat["model_names"][idx, 0][0]
            else:
                self.classes["model"][idx] = "NoName"

        for idx in range(mmmat["make_names"].shape[0]):  # --> 12
            if mmmat["make_names"][idx, 0].shape[0] > 0:
                self.classes["make"][idx] = mmmat["make_names"][idx, 0][0]
            else:
                self.classes["make"][idx] = "NoName"
        del mmmat
        del tmat

        # Get the Model-IDs to Car Type dictionary from the attributes file
        # attributes file --> (model_id maximum_speed displacement door_number seat_number type)
        model_id_type_dict = {}
        with open(
            os.path.join(os.path.join(self.data_folder, "misc"), "attributes.txt"), "r"
        ) as attr_file:
            attr_file.readline()
            for line in attr_file:
                dat = line.strip().split(" ")
                model_id_type_dict[dat[0]] = dat[-1]

        split_folder = os.path.join(self.data_folder, "train_test_split")
        class_folder = os.path.join(split_folder, "classification")

        (
            self.metadata["train"],
            self.metadata["test"],
            self.metadata["val"],
            self.metadata["full"],
        ) = ({}, {}, {}, {})
        self.metadata["train"]["crawl"], self.metadata["train"]["imgs"] = self.__crawl(
            class_folder, self.train_file, model_id_type_dict
        )
        self.metadata["test"]["crawl"], self.metadata["test"]["imgs"] = self.__crawl(
            class_folder, self.test_file, model_id_type_dict
        )
        self.metadata["val"]["crawl"], self.metadata["val"]["imgs"] = [], 0
        # full is usually handled in generator
        self.metadata["full"]["crawl"], self.metadata["full"]["imgs"] = [], 0

        # Here we need to perform some data cleaning
        # Namely, the make, model, and year ids are the raw make model
        # We need to convert them a bit, in that we need only the make model and year that exist, and have them be consistent across train and test
        self.existingmakes = {
            original: remapped
            for remapped, original in enumerate(
                set([item[self.makeidx] for item in self.metadata["train"]["crawl"]])
            )
        }
        self.existingmodels = {
            original: remapped
            for remapped, original in enumerate(
                set([item[self.modelidx] for item in self.metadata["train"]["crawl"]])
            )
        }
        self.existingyears = {
            original: remapped
            for remapped, original in enumerate(
                set([item[self.yearidx] for item in self.metadata["train"]["crawl"]])
            )
        }

        self.metadata["train"]["crawl"] = [
            [
                self.existingmakes[sampletuple[self.makeidx]],
                self.existingmodels[sampletuple[self.modelidx]],
                self.existingyears[sampletuple[self.yearidx]],
                sampletuple[self.typeidx],
                sampletuple[self.pathidx],
            ]
            for sampletuple in self.metadata["train"]["crawl"]
        ]
        self.metadata["test"]["crawl"] = [
            [
                self.existingmakes[sampletuple[self.makeidx]],
                self.existingmodels[sampletuple[self.modelidx]],
                self.existingyears[sampletuple[self.yearidx]],
                sampletuple[self.typeidx],
                sampletuple[self.pathidx],
            ]
            for sampletuple in self.metadata["test"]["crawl"]
        ]

        # Now is when we have the year information
        self.classes["year"] = self.existingyears

        # set up the necessary class information...
        for meta in ["train", "test", "val", "full"]:
            self.metadata[meta]["classes"] = {}
            self.metadata[meta]["classes"]["type"] = len(self.classes["type"])
            self.metadata[meta]["classes"]["model"] = len(self.existingmodels)
            self.metadata[meta]["classes"]["make"] = len(self.existingmakes)
            self.metadata[meta]["classes"]["year"] = len(self.classes["year"])

    def __crawl(self, folder, file, model_type_dict):

        data_tuple = (
            []
        )  # (makeid, modelid,releasedyear, type, str(path-to-imagename))   <-- EVERYTHING ELSE IS INT
        basepath = os.path.join(self.data_folder, "image")
        with open(os.path.join(folder, file), "r") as data_list:
            for line in data_list:
                parsed_line = line.strip().split("/")[:3]
                parsed_line.append(model_type_dict[parsed_line[1]])
                parsed_line = [
                    self.convert(item) - 1 for item in parsed_line
                ]  # WE RESET LABELS TO 0!!!!!!!!!!!! NOTE NOTE NOTE NOTE
                parsed_line[2] = parsed_line[2] + 1
                parsed_line.append(os.path.join(basepath, line.strip()))
                data_tuple.append(tuple(parsed_line))
        return data_tuple, len(data_tuple)
        # Now we calculate the number for each class

    def convert(self, val):
        if val == "unknown":  # For year...
            return -1
        return int(val)
