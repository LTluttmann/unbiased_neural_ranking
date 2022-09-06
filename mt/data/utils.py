import tensorflow as tf
import pandas as pd
import tempfile
import boto3
from mt.utils import ensure_url_format

def _bytes_feature(value_list):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value_list, type(tf.constant(0))):
        value_list = value_list.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def _float_feature(value_list):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _int64_feature(value_list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def get_blocklist(path):
    url = ensure_url_format(path)
    client = boto3.client("s3")
    with tempfile.NamedTemporaryFile() as f:
        client.download_file(url.bucket, url.key, f.name)
        block_list = pd.read_csv(f.name)
    block_list = block_list.iloc[:,0].values.tolist()
    return block_list