import logging
import os
from ednaml.logging import LogManager
from ednaml.utils import ERSKey


class FileLogManager(LogManager):
    logLevels = {
        0: logging.NOTSET,
        1: logging.ERROR,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    verbose: int = 3
    def apply(self, **kwargs):
        self.logger = logging.Logger(self.experiment_key.getExperimentName())
        self.buildLogger(
                self.logger,
                logger_given = False,   # For the log level setup
                add_filehandler=False,
                add_streamhandler=True,
                log_level = kwargs.get("log_level", logging.DEBUG))


    def updateERSKey(self, ers_key: ERSKey, file_name: str):
        self.buildLogger(
            self.logger,
            logger_given=True,
            add_filehandler=True,
            add_streamhandler=False,
            logger_save_path=file_name,
        )

    def buildLogger(
        self,
        logger: logging.Logger,
        logger_given = False,
        add_filehandler: bool = False,
        add_streamhandler: bool = True,
        logger_save_path: str = "",
        log_level = logging.DEBUG,
        **kwargs
    ) -> logging.Logger:
        """Builds a new logger or adds the correct file and stream handlers to
        existing logger if it does not already have them.

        Args:
            logger (logging.Logger, optional): A logger.. Defaults to None.
            add_filehandler (bool, optional): Whether to add a file handler to the logger. If False, no file is created or appended to. Defaults to True.
            add_streamhandler (bool, optional): Whether to add a stream handler to the logger. If False, logger will not stream to stdout. Defaults to True.

        Returns:
            logging.Logger: A logger with file and stream handlers.
        """
        
        streamhandler = False
        filehandler = False

        if logger.hasHandlers():
            for handler in logger.handlers():
                if isinstance(handler, logging.StreamHandler):
                    streamhandler = True
                if isinstance(handler, logging.FileHandler) and add_filehandler:
                    if os.path.splitext(os.path.basename(handler.baseFilename))[0] == os.path.splitext(os.path.basename(logger_save_path))[0]:
                        filehandler = True

        if not logger_given:
            logger.setLevel(self.logLevels[self.verbose])

        if not filehandler and add_filehandler:
            fh = logging.FileHandler(
                logger_save_path, mode="a", encoding="utf-8"
            )
            fh.setLevel(log_level)
            fh.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            logger.addHandler(fh)

        if not streamhandler and add_streamhandler:
            cs = logging.StreamHandler()
            cs.setLevel(log_level)
            cs.setFormatter(
                logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(cs)
        return None

        