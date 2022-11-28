from ednaml.utils import ERSKey, ExperimentKey



class LogManager:
    experiment_key: ExperimentKey
    def __init__(self, experiment_key: ExperimentKey, **kwargs):
        self.experiment_key = experiment_key
        self.logger = None
        self.apply(**kwargs)
    
    def apply(self, **kwargs):
        """Build the logger internal state. At this time, the logger does not have access 
        to the ERSKey, or any indexing information about the current experiment.

        This function can be used to initialize logging, and set of batched requests once
        the logger has access to indexing information.
        """
        raise NotImplementedError()

    def updateERSKey(self, ers_key: ERSKey, file_name: str):
        """Update the logger with the run information from the ERSKey. The StorageKey is also available
        if this logger indexes logs as such.

        The file_name is the local file where logs can be dumped. A Log Storage will upload this local file
        to its remote Storage with the latest ERS-Key. This means a batch of logs is generally indexed by the 
        logging backup frequency (depending on how the Log Storage takes care of files)

        Args:
            ers_key (ERSKey): _description_
            file_name (str): _description_
        """
        raise NotImplementedError()

    def getLogger(self):
        return self.logger

    def flush(self) -> bool:
        """Flush any remaining logs.

        Returns:
            bool: Flush success
        """
        pass

    def getLocalLog(self) -> str:
        """Return path to a local file containing any disk logs. Can be an empty file.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: Path to log file
        """
        raise NotImplementedError()