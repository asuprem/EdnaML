from abc import ABC

class BaseConfig(ABC):
    def getVars(self):
        return vars(self)