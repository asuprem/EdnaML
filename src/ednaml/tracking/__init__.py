



from logging import Logger
from os import PathLike
from typing import Dict
from ednaml.storage.BaseStorage import BaseStorage

class Backup:
    def __init__(self, storage_name: str, storage_frequency: int, storage: Dict[str,BaseStorage], logger: Logger):
        self.storage_name = storage_name
        self.storage_frequency = storage_frequency
        self.storage = storage
        self.initial_backup = False
        self.logger = logger

    def backup(self, file_name: PathLike, epoch: int):
        """Use storage to perform backup

        Args:
            file_name (PathLike): _description_
            epoch (int): _description_
        """
        if self.storage_frequency == -1:
            pass
        elif self.storage_frequency == 0 and not self.initial_backup and self.storage_name in self.storage:
            self.performBackup(file_name, epoch)
        elif self.storage_frequency is None or self.storage_frequency > 0 and self.storage_name in self.storage:
            self.performBackup(file_name, epoch)
        elif self.storage_name not in self.storage:
            self.logger.info("Not backing up {filename}. Storage name [{storage_name}] not in list of provided Storages.".format(
                filename = file_name,
                storage_name = self.storage_name
            ))
        else:
            pass

    def performBackup(self, file_name: PathLike, epoch: int):
        pass


class ConfigBackup(Backup):
    """_summary_

    Args:
        Backup (_type_): _description_
    """
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)

class ModelBackup(Backup):
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)

class ModelArtifactsBackup(Backup):
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)

class ModelPluginBackup(Backup):
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)

class MetricsBackup(Backup):
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)

class LogBackup(Backup):
    def performBackup(self, file_name: PathLike, epoch: int):
        self.storage[self.storage_name].save(file_path = file_name)


class BackupManager:
    configBackup: ConfigBackup
    modelBackup: ModelBackup
    modelArtifactsBackup: ModelArtifactsBackup
    modelPluginBackup: ModelPluginBackup
    metricsBackup: MetricsBackup
    logBackup: LogBackup

    def __init__(self, configbackup, modelbackup, modelartifactsbackup, modelpluginbackup, metricsbackup, logbackup):
        self.configBackup = configbackup
        self.modelBackup = modelbackup
        self.modelArtifactsBackup = modelartifactsbackup
        self.modelPluginsBackup = modelpluginbackup
        self.metricsBackup = metricsbackup
        self.logBackup = logbackup


