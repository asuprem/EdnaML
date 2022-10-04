from typing import Any, Dict, List
import ednaml
from ednaml.crawlers import Crawler
import ednaml.core.decorators as edna
import json, os
from ednaml.crawlers import Crawler
from ednaml.utils.web import download
from ednaml.utils.file_utils import IterableFile
import gzip
import shutil


# This is to read from Azure...download a thing from azure, and unzip, and load it into memory...
# We will datashard it with the Datareader args, specifically for the Deployment...


@edna.register_crawler
class FNCCrawler(Crawler):
    """Downloads a FNC raw file from Azure if it exists and builds an iterable from the file contents.

    Crawler yields a list of tweet objects, i.e. a dictionary containing tweet attributes.
    """
    def __init__(
        self,
        logger=None,
        azstorage="ednadatasets",
        azcontainer="edna-covid-raw",
        azfile="tweets-2020-01-22.json.gz",
        test_file = None,
        train_azstorage=None,
        train_azcontainer=None,
        train_azfile=None,
        train_file = None
    ):
        """Initializes the FNC Crawler.

        By default, only the test data is populated. If the corresponding values for arguments is None, it is not used, instead of throwing error.

        Args:
            logger (_type_, optional): Logger. Defaults to None.

            azstorage (str, optional): Axure storage name. Defaults to "ednadatasets".
            azcontainer (str, optional): Azure container name. Defaults to "edna-covid-raw".
            azfile (str, optional): Azure file name (blob name). Defaults to "tweets-2020-01-22.json.gz".
            test_file (_type_, optional): If using local files, this will supersede the Azure data. Defaults to None.

            train_azstorage (str, optional): Axure storage name. Defaults to "ednadatasets".
            train_azcontainer (str, optional): Azure container name. Defaults to "edna-covid-raw".
            train_azfile (str, optional): Azure file name (blob name). Defaults to "tweets-2020-01-22.json.gz".
            train_file (_type_, optional): If using local files, this will be present in the "train" metadata and supersede Azure data. Defaults to None.
            
        """
        if test_file is None:
            if azstorage is not None:
                az_url = self.build_url(azstorage, azcontainer, azfile)
                logger.info("Crawling %s" % (az_url))
                if not os.path.exists(azfile):
                    download(azfile, az_url)
                else:
                    logger.info("%s already exists at %s" % (az_url, azfile))
                az_jsonfile = os.path.splitext(azfile)[0]

                # Then unzip the file
                if not os.path.exists(az_jsonfile):
                    with gzip.open(azfile, "rb") as f_in:
                        with open(az_jsonfile, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
            else:
                az_jsonfile = None
        else:   # TODO possible sanity checks????
            az_jsonfile = test_file


        if train_file is None:
            if train_azstorage is not None:
                train_az_url = self.build_url(train_azstorage, train_azcontainer, train_azfile)
                logger.info("Crawling %s" % (train_az_url))
                if not os.path.exists(train_azfile):
                    download(train_azfile, train_az_url)
                else:
                    logger.info("%s already exists at %s" % (train_az_url, train_azfile))
                train_az_jsonfile = os.path.splitext(train_azfile)[0]

                # Then unzip the file
                if not os.path.exists(train_az_jsonfile):
                    with gzip.open(train_azfile, "rb") as f_in:
                        with open(train_az_jsonfile, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
            else:
                train_az_jsonfile = None
        else:   # TODO possible sanity checks????
            train_az_jsonfile = train_file

        # set up class metadata
        self.classes = {}
        self.classes["fnews"] = 2

        # set up content metadata
        self.metadata = {}
        self.metadata["secondary"] = {}

        if az_jsonfile is not None:
            self.metadata["secondary"]["linecount"] = self.bufcount(az_jsonfile)
        else:
            self.metadata["secondary"]["linecount"] = 0
        if train_az_jsonfile is not None:
            self.metadata["secondary"]["train_linecount"] = self.bufcount(train_az_jsonfile)
        else:
            self.metadata["secondary"]["train_linecount"] = 0

        self.metadata["train"] = {}
        self.metadata["test"] = {}
        self.metadata["val"] = {}
        self.metadata["train"]["crawl"] = []
        self.metadata["test"]["crawl"] = []
        self.metadata["val"]["crawl"] = []

        # Get the tweet data as list, <id, text, url> TODO
        # basically need a generator that yields results and maps them through some function
        # This will be slow because of json.loads line by line. We should speed this up with chunking?

        # TODO extracts full_text only. Fix this so it extracts all relevant objects (or possibly all objects??)
        if az_jsonfile is not None:
            self.metadata["test"]["crawl"] = IterableFile(
                az_jsonfile,
                line_callback=lambda row: (json.loads(row.strip())["full_text"], 0,0,0),
            )
        if train_az_jsonfile is not None:
            self.metadata["train"]["crawl"] = IterableFile(
                train_az_jsonfile,
                line_callback=lambda row: (json.loads(row.strip())["full_text"], 0,0,0),
            )
        self.metadata["train"]["classes"] = self.classes
        self.metadata["test"]["classes"] = self.classes
        self.metadata["val"]["classes"] = self.classes
        # pdb.set_trace()

    def build_url(self, azstorage, azcontainer, azfile):
        return "https://{azstorage}.blob.core.windows.net/{azcontainer}/{azfile}".format(
            azstorage=azstorage, azcontainer=azcontainer, azfile=azfile
        )

    def bufcount(self, filename):
        f = open(filename)                  
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read # loop optimization

        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)

        return lines