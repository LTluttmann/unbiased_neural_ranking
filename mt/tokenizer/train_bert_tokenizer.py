import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner
from tensorflow_text.python.ops import bert_tokenizer

import boto3
import os
from typing import List

from mt.utils import S3Url, get_latest_subfolder_from_url
from mt.data.dataset.parser import ExampleParser
from mt.tokenizer.tokenizer_io import get_vocab_to_s3
from mt.config import config

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=config.VOCAB_SIZE,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

s3url = S3Url(config.MT_OUTPUT_URL)
MT_OUTPUT_BUCKET = s3url.bucket


def bert_vocab_from_dataset(dataset,
                            vocab_size: int,
                            reserved_tokens: List[str],
                            bert_tokenizer_params=None,
                            learn_params=None,
                            preserve_accents=True) -> List[str]:

    if bert_tokenizer_params is None:
        bert_tokenizer_params = {}
    if learn_params is None:
        learn_params = {}
    element_spec = dataset.element_spec
    try:
        element_spec.shape
    except AttributeError:
        raise TypeError("The dataset should contain single-tensor elements.")

    tokenizer = bert_tokenizer.BasicTokenizer(**bert_tokenizer_params)
    if preserve_accents:
        tokenizer = bert_tokenizer.AccentPreservingBasicTokenizer(
            **bert_tokenizer_params)
    else:
        tokenizer = bert_tokenizer.BasicTokenizer(**bert_tokenizer_params)
    words_dataset = dataset.map(tokenizer.tokenize)
    word_counts = learner.count_words(words_dataset)
    vocab = learner.learn(word_counts, vocab_size, reserved_tokens,
                          **learn_params)

    return vocab


def main(file_pattern, limit:int = None, merge_vocabs: bool = True):
    client = boto3.client("s3")
    s3url = S3Url(config.MT_DR_BANNER_URL)
    project_folder_key = get_latest_subfolder_from_url(client, s3url, config.TIMESTAMP_SIGNATURE_FORMAT)

    feature_map = {col: tf.io.FixedLenFeature([], tf.string) for col in config.BERT_VOCABS}
    parser = ExampleParser(feature_map, on_batch=False)

    dataset = (
        tf.data.Dataset
        .list_files(file_pattern, shuffle=True)
        .interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.AUTOTUNE)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    )
    if limit:
        dataset = dataset.take(limit)

    datasets = {
        col: dataset.map(lambda x: x[col])
        for col in config.BERT_VOCABS
    }

    if merge_vocabs:
        filename = config.BERT_VOCAB_FILENAME
        # merge dataset
        merged_ds = datasets.pop(config.BERT_VOCABS[0])
        for ds in datasets.values():
            merged_ds = merged_ds.concatenate(ds)
        # generate vocab for dataset
        vocab = bert_vocab_from_dataset(
            merged_ds.batch(10_000).prefetch(tf.data.AUTOTUNE),
            **bert_vocab_args
        )
        vocab_file_key = os.path.join(s3url.key, project_folder_key, config.BERT_VOCAB_SUFFIX, filename)
        get_vocab_to_s3(vocab, client, s3url.bucket, vocab_file_key)

    else:
        # iterate through datasets
        for col, ds in datasets.items():
            filename = f"{col}_vocab.txt"
            vocab = bert_vocab_from_dataset(
                ds.batch(10_000).prefetch(tf.data.AUTOTUNE),
                **bert_vocab_args
            )
            vocab_file_key = os.path.join(s3url.key, project_folder_key, config.BERT_VOCAB_SUFFIX, filename)
            get_vocab_to_s3(vocab, client, s3url.bucket, vocab_file_key)


if __name__ == "__main__":
    main()
