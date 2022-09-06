from datetime import datetime, timedelta
import os
import shutil
import boto3
from urllib.parse import urlparse
import logging
import sys


class S3Url(object):

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()


def get_latest_subfolder_from_url(client, s3url: S3Url, timestamp_format):
    folders = client.list_objects(
        Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/')
    max_timestamp = datetime.now()-timedelta(days=1000)
    for folder in folders.get("CommonPrefixes"):
        timestamp = folder.get("Prefix").split("/")[len(s3url.key.split("/"))-1]
        try:
            datetime_object = datetime.strptime(timestamp, timestamp_format)
        except ValueError:
            continue
        if datetime_object > max_timestamp:
            most_recent_folder = folder.get("Prefix")
    most_recent_folder = os.path.split(most_recent_folder.rstrip("/"))[-1]
    return most_recent_folder


def s3_folder_exists_and_not_empty(client, path:str) -> bool:
    '''
    Folder should exists. 
    Folder should not be empty.
    '''
    if not path.endswith('/'):
        path = path + "/"
    s3url = S3Url(path)
    resp = client.list_objects(
        Bucket=s3url.bucket, Prefix=s3url.key, MaxKeys=1)
    return 'Contents' in resp


def download_dir(client, bucket, path, target, preserve_subfolders=True):
    # Handle missing / at end of prefix
    if not path.endswith('/'):
        path += '/'

    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=path):
        # Download each file individually
        for key in result['Contents']:
            # Calculate relative path
            rel_path = key['Key'][len(path):]
            # Skip paths ending in /
            if not key['Key'].endswith('/'):
                if preserve_subfolders:
                    local_file_path = os.path.join(target, rel_path)
                    local_file_dir = os.path.dirname(local_file_path)
                else:
                    local_file_dir = target
                    local_file_path = os.path.join(
                        target, rel_path.replace("/", "_"))
                # Make sure directories exist
                ensure_dirs_exists(local_file_dir)
                client.download_file(bucket, key['Key'], local_file_path)


def download_all_tfrecords_from_s3(
    url: str, destination: str, timestamp_format: str,
    tfrecord_prefix: str = "tfrecord_data",
    preserve_subfolders=True
):
    if not url.endswith('/'):
        url += '/'
    client = boto3.client("s3")
    s3url = S3Url(url)
    most_recent_folder = get_latest_subfolder_from_url(
        client, s3url, timestamp_format)
    key = os.path.join(s3url.key, most_recent_folder, tfrecord_prefix)
    download_dir(client, s3url.bucket, key, destination,
                 preserve_subfolders=preserve_subfolders)

    return most_recent_folder


def ensure_url_format(url):
    if not isinstance(url, S3Url):
        return S3Url(url)
    else:
        return url


def downloadDirectoryFroms3(bucket, key, local_file_name):
    s3 = boto3.client('s3')
    with open(local_file_name, 'wb') as f:
        s3.download_fileobj(bucket, key, f)


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def ensure_dirs_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def make_or_delete_dir(dir):
    delete_dir(dir)
    os.mkdir(dir)


def ensure_proper_s3_folder_path(url: str):
    return url.rstrip("/").__add__("/")


def create_logger():
    logger = logging.getLogger('mt')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    return logger

