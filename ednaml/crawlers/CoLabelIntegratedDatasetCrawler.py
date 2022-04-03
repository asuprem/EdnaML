import os


class CoLabelIntegratedDatasetCrawler:
    """Data crawler for CoLabel Integrated Data dataset
    """

    def __init__(
        self,
        data_folder="CoDataset",
        train_folder="image",
        test_folder="",
        validation_folder="",
        **kwargs
    ):
        self.metadata = {}
        self.classes = {}
        self.annotations = {}
        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)

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
        human_readable_directory = os.path.join(self.data_folder, "readable")

        # Set up type human readable
        annotations = ["make", "color", "type"]
        for annotation in annotations:
            self.extract_readable_class_names(
                annotation, human_readable_directory, self.classes
            )  # into self.classes...
            self.extract_readable_class_names(
                annotation, human_readable_directory, self.annotations
            )  # into self.classes...
        self.extract_readable_class_names(
            "model", human_readable_directory, self.classes
        )  # into self.classes...

        split_folder = os.path.join(self.data_folder, "splits")
        class_folder = os.path.join(split_folder, "vmmr")

        self.metadata["train"], self.metadata["test"], self.metadata["val"] = {}, {}, {}
        self.metadata["train"]["crawl"], self.metadata["train"]["imgs"] = self.__crawl(
            class_folder, "train.txt"
        )
        self.metadata["test"]["crawl"], self.metadata["test"]["imgs"] = self.__crawl(
            class_folder, "test'txt"
        )
        # self.metadata["val"]["crawl"], self.metadata["val"]["classes"], self.metadata["val"]["imgs"] = self.__crawl(self.val_folder)

        self.metadata["train"]["classes"] = {}
        self.metadata["train"]["classes"]["type"] = len(self.classes["type"])
        self.metadata["train"]["classes"]["model"] = len(self.classes["model"])
        self.metadata["train"]["classes"]["make"] = len(self.classes["make"])
        self.metadata["train"]["classes"]["color"] = len(self.classes["color"])
        self.metadata["test"]["classes"] = {}
        self.metadata["test"]["classes"]["type"] = len(self.classes["type"])
        self.metadata["test"]["classes"]["model"] = len(self.classes["model"])
        self.metadata["test"]["classes"]["make"] = len(self.classes["make"])
        self.metadata["test"]["classes"]["color"] = len(self.classes["color"])

    def extract_readable_class_names(self, annotation, dir, storage):
        """Extracts the class names for make, model, etc from the respective txt files. Each line
        corresponds to a class id, and the string is the human-readable format.

        Args:
            annotation (str): The annotation to retrieve. It is in the [annotation].txt file
            dir (str): The directory where these are kept
            storage (Dict): Which dict to store these annotations
        """
        idx = 0
        storage[annotation] = {}
        with open(os.path.join(dir, annotation + ".txt"), "r") as ofile:
            for line in ofile:
                storage[annotation][idx] = line.strip()
            idx += 1

    def __crawl(self, folder, file, model_type_dict):

        data_tuple = (
            []
        )  # (makeid, modelid,typeid, colorid, str(imagename))   <-- EVERYTHING ELSE IS INT
        basepath = os.path.join(self.data_folder, "image")
        with open(os.path.join(folder, file), "r") as data_list:
            for line in data_list:
                parsed_line = line.strip().split("/")[
                    :4
                ]  # <type>/<color>/[78/1/3ac218c0c6c378.jpg] --> [typeid,colorid,makeid,modelid,path.jpg]
                parsed_line = [
                    int(item) - 1 for item in parsed_line
                ]  # WE RESET LABELS TO 0!!!!!!!!!!!! NOTE NOTE NOTE NOTE
                parsed_line.append(
                    os.path.join(basepath, "/".join(parsed_line[2:]))
                )  #  makeid,modelid,path.jpg --> makeid/modelid/path.jpg
                data_tuple.append(tuple(parsed_line))
        return data_tuple, len(data_tuple)
        # Now we calculate the number for each class
