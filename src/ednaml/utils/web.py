from urllib.parse import urlparse
import sys, requests
import sys
import json
import os
import shutil
import tempfile
import fnmatch
from functools import wraps
from hashlib import sha256
from io import open
from tqdm import tqdm

try:
    import boto3
    from botocore.config import Config as botoConfig
    from botocore.exceptions import ClientError as botoClientError
except ImportError:
    boto3 = None
    botoConfig = None
    botoClientError = None

# from https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(file_, url):
    sys.stdout.write("Downloading %s from %s" % (file_, url))
    with open(file_, "wb") as f:
        response = requests.get(url, stream=True)
        all_data = response.headers.get("content-length")
        if all_data is None:
            f.write(response.content)
        else:
            all_data = int(all_data)
            progress = 0
            for chunk in response.iter_content(
                chunk_size=max(int(all_data / 1000), 1024 * 1024)
            ):
                progress += len(chunk)
                f.write(chunk)
                sys.stdout.write(
                    "\r{}/{} bytes [{}{}]".format(
                        all_data,
                        progress,
                        "█" * int(100 * progress / all_data),
                        "." * (100 - int(100 * progress / all_data)),
                    )
                )
                sys.stdout.flush()
    sys.stdout.write("\nDownload of %s to %s completed\n" % (file_, url))


def cached_path(
    url_or_filename, cache_dir=None, force_download=False, proxies=None
):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
    """
    import os
    from torch.hub import _get_torch_home
    from pathlib import Path

    if cache_dir is None:
        cache_dir = os.getenv(
            "PYTORCH_TRANSFORMERS_CACHE",
            os.getenv(
                "PYTORCH_PRETRAINED_BERT_CACHE",
                os.path.join(_get_torch_home(), "pytorch_transformers"),
            ),
        )
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(
                url_or_filename
            )
        )


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    import os
    from torch.hub import _get_torch_home
    from pathlib import Path

    if cache_dir is None:
        cache_dir = os.getenv(
            "PYTORCH_TRANSFORMERS_CACHE",
            os.getenv(
                "PYTORCH_PRETRAINED_BERT_CACHE",
                os.path.join(_get_torch_home(), "pytorch_transformers"),
            ),
        )
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except EnvironmentError:
            etag = None

    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode("utf-8")
    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + ".*")
        matching_files = list(
            filter(lambda s: not s.endswith(".json"), matching_files)
        )
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    if not os.path.exists(cache_path) or force_download:
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            print(
                "%s not found in cache or force_download set to True,"
                " downloading to %s",
                url,
                temp_file.name,
            )

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            print("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            print("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w") as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(
                        output_string, "utf-8"
                    )  # The beauty of python 2
                meta_file.write(output_string)

            print("removing temp file %s", temp_file.name)

    return cache_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except botoClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3", config=botoConfig(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3", config=botoConfig(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path
