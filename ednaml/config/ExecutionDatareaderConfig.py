



from typing import Dict


class ExecutionDatareaderConfig:
    DATAREADER: str
    CRAWLER_ARGS: Dict[str,str]
    DATASET_ARGS: Dict[str,str]
    GENERATOR: str
    GENERATOR_ARGS: Dict[str,str]
    DATALOADER_ARGS: Dict[str,str]


    def __init__(self, datareader_dict):
        self.DATAREADER = datareader_dict.get("DATAREADER")
        self.CRAWLER_ARGS = datareader_dict.get("CRAWLER_ARGS", {})
        self.DATASET_ARGS = datareader_dict.get("DATASET_ARGS", {})
        self.GENERATOR = datareader_dict.get("GENERATOR", "ClassificationGenerator")
        self.GENERATOR_ARGS = datareader_dict.get("GENERATOR_ARGS", {})
        self.DATALOADER_ARGS = datareader_dict.get("DATALOADER_ARGS", {})