import tensorflow_text as text

from mt.utils import S3Url, ensure_url_format

import boto3
import tempfile


def get_vocab_to_s3(vocab, client, s3_bucket:str, s3_key:str):
    with tempfile.NamedTemporaryFile("w") as f:
        for token in vocab:
            print(token, file=f)
        f.seek(0)
        client.upload_file(f.name, s3_bucket, s3_key)
    # TODO possibly makes sense to also generate tokenizer here and save it
    # saving tokenizer does not work: loaded tokenizer throws weird errors


def read_vocab(filename):
    with open(filename, "r") as f:
        vocab = f.read().splitlines()
    return vocab


def get_vocab_from_s3(path, client=None):
    path = ensure_url_format(path)
    if not client:
        client = boto3.client("s3")
    with tempfile.NamedTemporaryFile("w+") as f:
        client.download_file(path.bucket, path.key, f.name)
        f.seek(0)
        vocab = read_vocab(f.name)
        return vocab


def load_bert_tokenizer_from_vocab_path(path: S3Url, return_vocab:bool=False, **kwargs):
    path = ensure_url_format(path)
    with tempfile.NamedTemporaryFile("w+") as f:
        client = boto3.client("s3")
        client.download_file(path.bucket, path.key, f.name)
        f.seek(0)
        tokenizer = text.BertTokenizer(f.name, **kwargs)
        if return_vocab:
            vocab = read_vocab(f.name)
            return tokenizer, vocab
        else:
            return tokenizer