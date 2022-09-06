import tensorflow as tf

from mt.models.ltr.attnrank import AttnRank
from mt.utils import ensure_url_format

import boto3
import tempfile
import zipfile
import os


def zipdir(path, ziph):
    # Zipfile hook to zip up model folders
    length = len(path) # Doing this to get rid of parent folders
    for root, _, files in os.walk(path):
        folder = root[length:] # We don't need parent folders! Why in the world does zipfile zip the whole tree??
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))


def s3_save_keras_model(model, model_name, bucket, key, client=None):
    with tempfile.TemporaryDirectory() as tempdir:
        tf.saved_model.save(model, f"{tempdir}/{model_name}")
        # Zip it up first
        zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
        zipdir(f"{tempdir}/{model_name}", zipf)
        zipf.close()
        if client == None:
            client = boto3.client("s3")
        client.upload_file(
            f"{tempdir}/{model_name}.zip", Bucket=bucket, Key=os.path.join(key, model_name+".zip")
        )        


def copy_keras_model_to_s3(filepath, bucket, key, model_name, client=None):
    with tempfile.NamedTemporaryFile() as f:
        # Zip it up first
        zipf = zipfile.ZipFile(f.name, "w", zipfile.ZIP_STORED)
        zipdir(filepath, zipf)
        zipf.close()
        if client == None:
            client = boto3.client("s3")
        client.upload_file(
            f.name, Bucket=bucket, Key=os.path.join(key, model_name+".zip")
        )   

def copy_keras_weights_to_s3(filepath, bucket, key, model_name, client=None):
    if client == None:
        client = boto3.client("s3")
    client.upload_file(
        filepath, Bucket=bucket, Key=os.path.join(key, model_name+".h5")
    )   


def s3_save_keras_weights(model, filename, bucket, key, client=None):
    with tempfile.NamedTemporaryFile() as f:
        model.save_weights(f.name, save_format="h5")
        if client == None:
            client = boto3.client("s3")
        client.upload_file(
            f.name, Bucket=bucket, Key=os.path.join(key, filename + ".h5")
        )


def s3_get_keras_model_from_weights(model, bucket, key, client=None):
    with tempfile.NamedTemporaryFile() as f:
        if client == None:
            client = boto3.client("s3")
        client.download_file(Bucket=bucket, Key=key, Filename=f.name)
        model.load_weights(f.name)


def s3_get_keras_model(model_name: str, s3_url:str) -> tf.keras.Model:
    with tempfile.TemporaryDirectory() as tempdir:
        # Fetch and save the zip file to the temporary directory
        client = boto3.client("s3")
        client.download_file(s3_url.bucket, os.path.join(s3_url.key, model_name)+".zip", f"{tempdir}/{model_name}.zip")
        # Extract the model zip file within the temporary directory
        with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
            zip_ref.extractall(f"{tempdir}/{model_name}")
        # Load the keras model from the temporary directory
        return tf.keras.models.load_model(f"{tempdir}/{model_name}")

def load_and_init_ranker_model(model_name, model_url, example_data):
    """this utility function loads a saved keras model from s3, saves its weights and
    initializes a new object with the same class and weights as the loaded model. The 
    reason for this function is that keras has problems to apply a saved and loaded
    model on data of different shape that it was trained on. Using this approach is
    hacky, but solves these issues."""

    model_url = ensure_url_format(model_url)
    model = s3_get_keras_model(model_name, model_url)

    model_class = eval(type(model).__name__)

    config = model.get_config()
    final_model = model_class(**config)
    _ = final_model(example_data)
    with tempfile.NamedTemporaryFile() as f:
        model.save_weights(f.name, save_format="h5")
        final_model.load_weights(f.name)
    return final_model