from ctypes import Union
from typing import Dict, List


class LabelMetadata:
    """LabelMetadata contains metadata about labels in the dataset.

    Attributes:
        labels (List[str]): A list of label names.
        totalLabels (int): The total number of labels in the LabelMetadata
        metadata (Dict[str,Dict[str,Union[str,int]]]): A dictionary of label-names to their metadata. Currently, stores the number of classes in metadata[labelname]["classes"]
    """
    labels : List[str]
    totalLabels : int
    metadata: Dict[str,Dict[str,int]]
        
    def __init__(self, label_dict: Dict[str,Dict[str,int]] = {}):
        """Initializes the LabelMetadata and populates it. If given an empty dict as input, the LabelMetadata will be empty

        Args:
            label_dict (Dict[str,Dict[str,Union[str,int]]], optional): A dictionary of label-names to their metadata, stored as label_dict[labelname]["classes"]. Defaults to {}.
        """
        self.labels = [*label_dict]
        self.totalLabels = len(label_dict)

        self.metadata = {labelname: {
                                "classes": label_dict[labelname]["classes"],
                            } 
                            for labelname in self.labels}


    def getLabels(self) -> List[str]:
        """Get the list of labels stored in this LabelMetadata

        Returns:
            List[str]: List of label names
        """
        return self.labels

    def getLabel(self, idx) -> str:
        """Get the label stored at this index

        Returns:
            str: Label names
        """
        return self.labels[idx]

    def getLabelDimensions(self, label:str=None) -> int:
        """Get the number of classes for a label.

        Args:
            label (str, optional): The label to retrieve number of classes for. If none, gets the first label's classes. Defaults to None.

        Returns:
            int: The number of classes
        """
        
        if label is None:
            return self.metadata[self.labels[0]]["classes"]
        return self.metadata[label]["classes"]

    def getTotalLabels(self)->int:
        """Gets the total number of labels stored in this LabelMetadata

        Returns:
            int: Total number of labels
        """
        return self.totalLabels

    def addLabel(self, label:str):
        if label in self.metadata:
            raise KeyError("Label %s already in class"%label)
        self.metadata[label] = {}