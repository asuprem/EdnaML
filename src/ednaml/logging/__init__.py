from ednaml.utils import ERSKey, ExperimentKey



class LogManager:
    experiment_key: ExperimentKey
    def __init__(self, experiment_key: ExperimentKey, **kwargs):
        self.experiment_key = experiment_key
        self.logger = None
        self.apply(**kwargs)
    
    def apply(self, **kwargs):
        pass

    def updateERSKey(self, ers_key: ERSKey, file_name: str):
        pass

    def getLogger(self):
        pass