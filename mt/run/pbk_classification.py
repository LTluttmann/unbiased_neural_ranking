from mt.data.dataset.callbacks.tokenizer_callback import TokenizerCallback
from mt.data.preprocessing.big_query import get_big_query_credentials, get_data_from_big_query
from mt.tokenizer.tokenizer_io import load_bert_tokenizer_from_vocab_path, read_vocab
from mt.utils import S3Url, create_logger, ensure_dir_exists
from mt.config import config
from mt.models.lse.pbk import PBKClassifier
from mt.models.model_io import s3_get_keras_model, s3_save_keras_weights
from mt.models.callbacks import SaveModelCallback

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

import boto3
import tensorflow as tf
import tensorflow.keras as nn
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split
import click
import numpy as np
from itertools import repeat
import os
import sys
import pickle


NORMED_WINS_COL = "normed_wins"
PATH_TRAIN = "./pbk_train"
PATH_VAL = "./pbk_val"

def get_datasets_for_single_label(spark, pbk_lookup, pbks, tokenizer, query, test_frac=0.1, batch_size=128, max_tokens=None):
    df = get_data_from_big_query(spark, get_big_query_credentials(), query)
    df = df.na.drop()

    df_normed = config.NORMALIZER(df, "tokens")
    df_normed = df_normed.filter(F.col("tokens") != "")
    pdf = df_normed.toPandas()

    query = pdf.tokens.values
    # label = np.eye(len(pbks)+1)[pbk_lookup(pdf.product_pd_AttributePbk).numpy()]
    label = tf.one_hot(pbk_lookup(pdf.product_pd_AttributePbk), depth=len(pbks)+1).numpy()

    queries_train, queries_val, labels_train, labels_val = train_test_split(query, label, test_size=test_frac)
    train_ds = get_dataset_from_queries_and_labels(queries_train, labels_train, tokenizer, batch_size, max_tokens)
    val_ds = get_dataset_from_queries_and_labels(queries_val, labels_val, tokenizer, batch_size, max_tokens)


    return train_ds, val_ds


def func(pdf, pbks):
    queries, labels = [], []
    for _, x in tqdm(pdf.iterrows(), total=pdf.shape[0]):
        mapping = [pbks.index(pbk)+1 if pbk in pbks else 0 for pbk in x[config.PBK_COL]]
        one_hot = np.zeros((len(pbks)+1,))
        one_hot[mapping] = x[NORMED_WINS_COL]
        queries.append(x[config.QUERY_COL])
        labels.append(one_hot)
    return queries, labels


def get_queries_and_labels(pdf, pbk_lookup, pbks, parallel=False, save_path=None):
    if save_path:
        if os.path.exists(save_path):
            queries = np.load(os.path.join(save_path, "queries.npy"), allow_pickle=True)
            labels = np.load(os.path.join(save_path, "labels.npy"), allow_pickle=True)
            return queries, labels

    if parallel:
        # create as many processes as there are CPUs on your machine
        num_processes = multiprocessing.cpu_count()

        # calculate the chunk size as an integer
        chunk_size = int(pdf.shape[0]/num_processes)

        # this solution was reworked from the above link.
        # will work even if the length of the dataframe is not evenly divisible by num_processes
        chunks = [pdf.iloc[pdf.index[i:i + chunk_size]] for i in range(0, pdf.shape[0], chunk_size)]

        # create our pool with `num_processes` processes
        pool = multiprocessing.Pool(processes=num_processes)

        # apply our function to each chunk in the list
        result = pool.starmap(func, zip(chunks, repeat(pbks)))

        queries, labels = zip(*result)

        queries = [item for sublist in queries for item in sublist]
        labels = [item for sublist in labels for item in sublist]

        if save_path:
            ensure_dir_exists(save_path)
            np.save(os.path.join(save_path, "queries.npy"), queries, allow_pickle=True)
            np.save(os.path.join(save_path, "labels.npy"), labels, allow_pickle=True)

    else:
        queries, labels = [], []
        for _, x in tqdm(pdf.iterrows(), total=pdf.shape[0]):
            one_hot = tf.one_hot(pbk_lookup(x[config.PBK_COL]), depth=len(pbks)+1)
            wins = tf.reshape(tf.constant(x[NORMED_WINS_COL]), (-1,1))
            label = tf.reduce_sum(one_hot*wins, axis=0)
            # query = tokenizer.tokenize(x[config.QUERY_COL]).merge_dims(-2,-1)
            queries.append(x[config.QUERY_COL])
            labels.append(label.numpy())
    return queries, labels


def get_data_for_multilabel(spark,
                            pbk_lookup,
                            tokenizer,
                            query: str,
                            pbks: list,
                            logger,
                            test_frac=0.1,
                            batch_size=128,
                            max_tokens=None,
                            parallel=True):

    logger.info(f"execute query: \n {query}")
    df = get_data_from_big_query(spark, get_big_query_credentials(), query)
    df = df.na.drop()
    df = config.NORMALIZER(df, config.QUERY_COL).distinct()
    df = df.filter(F.col(config.QUERY_COL) != "")
    logger.info(f"got {df.count()} records. Prepare dataframe...")

    df_agg = df.groupby([config.QUERY_COL, config.PBK_COL]).agg(F.sum(F.col("wins")).alias("wins"))
    df_agg = df_agg.join(df_agg.groupby(config.QUERY_COL).agg(F.sum(F.col("wins")).alias("searchterm_occurences")), on=config.QUERY_COL)
    df_normed = df_agg.withColumn(NORMED_WINS_COL, F.col("wins") / F.col("searchterm_occurences")).select(config.QUERY_COL, config.PBK_COL, NORMED_WINS_COL)

    df_collect = (
        df_normed
        .groupBy(config.QUERY_COL)
        .agg(F.collect_list(config.PBK_COL).alias(config.PBK_COL),
            F.collect_list(NORMED_WINS_COL).alias(NORMED_WINS_COL))
    )

    logger.info(f"got {df_collect.count()} distinct searchterms. Prepare tensorflow datasets...")
    pdf = df_collect.toPandas()
    train_df, val_df = train_test_split(pdf, test_size=test_frac)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)
    
    queries_train, labels_train = get_queries_and_labels(train_df, pbk_lookup=pbk_lookup, pbks=pbks, parallel=parallel, save_path=PATH_TRAIN)
    #test(queries_train, labels_train, train_df, pbk_lookup, 3)
    
    queries_val, labels_val = get_queries_and_labels(val_df, pbk_lookup=pbk_lookup, pbks=pbks, parallel=parallel, save_path=PATH_VAL)
    #test(queries_val, labels_val, val_df, pbk_lookup, 3)
    
    train_ds = get_dataset_from_queries_and_labels(queries_train, labels_train, tokenizer, batch_size, max_tokens)
    val_ds = get_dataset_from_queries_and_labels(queries_val, labels_val, tokenizer, batch_size, max_tokens)
    logger.info(f"finished dataset preparation")
    
    return train_ds, val_ds


def get_dataset_from_queries_and_labels(queries, labels, tokenizer, batch_size, max_tokens=None):
    if tf.config.list_physical_devices('GPU'):
        with tf.device("cpu"):
            dataset = tf.data.Dataset.from_tensor_slices({"query": queries, "label": labels})
    else:
        dataset = tf.data.Dataset.from_tensor_slices({"query": queries, "label": labels})

    if not max_tokens:
        seq_length=None
    else:
        seq_length={"query": max_tokens}

    token_ops = TokenizerCallback(
        tokenizer=tokenizer, 
        cols=["query"],
        max_length=seq_length
    )

    def split(x):
        return x["query"], x["label"]

    ds = (
        dataset
        .shuffle(10_000)
        .batch(batch_size)
        .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
        .map(split, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds

def test(queries, labels, df, pbk_lookup, testruns):
    for _ in range(testruns):
        idx = np.random.choice(list(range(len(queries))))
        q = queries[idx]
        y = labels[idx]
        query_df = df[df.searchterm == q]
        for pbk, true in zip(query_df.product_pd_AttributePbk.values.tolist()[0], query_df.normed_wins.values.tolist()[0]):
            enc = pbk_lookup(pbk)
            actual = y[enc.numpy().tolist()]
            assert np.isclose(actual, true), f"true: {true}, actual: {actual}"


def train(spark,
          logger,
          encoder: str,
          multi_label=True,
          test_frac=0.1,
          batch_size=128,
          epochs=10,
          learning_rate=1e-3,
          learning_rate_finetune=1e-4,
          dropout_rate=0.5,
          batch_norm=True,
          input_batch_norm=False,
          max_tokens=None,
          parallel=False):

    models_url = S3Url(os.path.join(config.MT_DR_BANNER_URL, "20220404-14:40:51/model/"))
    vocabs_url = os.path.join(config.MT_OUTPUT_URL, "20220404-14:40:51/vocab")
    s3_query_vocab_url = S3Url(os.path.join(vocabs_url, "product_pd_Name_searchterm_vocab.txt"))
    pbk_vocab_url = S3Url(os.path.join(vocabs_url, "pbks.txt"))

    logger.info(f"Get pbks...")
    client = boto3.client("s3")
    client.download_file(pbk_vocab_url.bucket, pbk_vocab_url.key, "pbks.txt")
    pbks = read_vocab("pbks.txt")
    logger.info(f"Got {len(pbks)} PBKs")
    pbk_lookup = nn.layers.StringLookup(vocabulary=pbks)

    logger.info("load tokenizer and vocabulary...")
    tokenizer, vocab = load_bert_tokenizer_from_vocab_path(s3_query_vocab_url, return_vocab=True)
    logger.info(f"size of vocabulary is {len(vocab)}")

    if multi_label:
        logger.info("get data for multi label...")
        query_multi = "SELECT DISTINCT * FROM `XXXXXX`"
        train_ds, val_ds = get_data_for_multilabel(
            spark=spark, 
            pbk_lookup=pbk_lookup, 
            tokenizer=tokenizer, 
            query=query_multi, 
            pbks=pbks,
            logger=logger,
            test_frac=test_frac, 
            batch_size=batch_size, 
            max_tokens=max_tokens, 
            parallel=parallel)
        metrics=[]
    else:
        logger.info("get data for single label...")
        query_single = "SELECT searchterm AS tokens, product_pd_AttributePbk FROM `XXXXXXXX`"
        query_single = "SELECT product_pd_Name AS tokens, product_pd_AttributePbk FROM `XXXXXXXX` LIMIT 200000"
        
        train_ds, val_ds = get_datasets_for_single_label(
            spark=spark, 
            pbk_lookup=pbk_lookup, 
            pbks=pbks,
            tokenizer=tokenizer, 
            query=query_single, 
            test_frac=test_frac, 
            batch_size=batch_size, 
            max_tokens=max_tokens)

        metrics=["accuracy"]

    logger.info(f"load encoder {encoder}")
    encoder = s3_get_keras_model(encoder, models_url)
    
    classifier = PBKClassifier(
        encoder, pbks, dropout_rate=dropout_rate, 
        input_batch_norm=input_batch_norm, use_batch_norm=batch_norm, finetune=False)

    optimizer = nn.optimizers.Adam(learning_rate=learning_rate)

    classifier_name = f"{classifier.name}_{encoder.name}"
    
    # define callbacks
    save_model = SaveModelCallback(classifier_name, model_name=classifier_name, s3_path=models_url, save_best_only=True)
    early_stop = nn.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3, restore_best_weights=True)
    model1_callbacks = [early_stop]
    model2_callbacks = [early_stop, save_model]
    lr_callback = nn.callbacks.ReduceLROnPlateau(patience=2, verbose=1)
    if not isinstance(optimizer.learning_rate, nn.optimizers.schedules.LearningRateSchedule):
        model1_callbacks.append(lr_callback)
    model2_callbacks.append(lr_callback)

    # train model with encoder fixed
    classifier.compile(optimizer=optimizer,
                       loss=nn.losses.CategoricalCrossentropy(from_logits=False),
                       metrics=metrics)
    logger.info(f"train with encoder freezed")
    history = classifier.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # write history
    with open(f'{classifier_name}_base_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    # finetune
    classifier.encoder.trainable = True
    classifier.compile(optimizer=nn.optimizers.Adam(learning_rate=learning_rate_finetune),
                       loss=nn.losses.CategoricalCrossentropy(from_logits=False),
                       metrics=metrics)
    logger.info(f"train with encoder unfreezed")

    # write history
    history_fine = classifier.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=model2_callbacks)
    with open(f'{classifier_name}_finetune_history', 'wb') as file_pi:
        pickle.dump(history_fine.history, file_pi)


@click.command()
@click.option("-e", "--encoder", type=str, required=True)
@click.option("-t", "--test_frac", type=float, default=0.1)
@click.option("-bs", "--batch_size", type=int, default=128)
@click.option("-ep", "--epochs", type=int, default=20)
@click.option("-lr", "--learning_rate", type=float, default=1e-3)
@click.option("-lr2", "--learning_rate_finetune", type=float, default=1e-4)
@click.option("-dr", "--dropout_rate", type=float, default=0.5)
@click.option("-bn", "--batch_norm", is_flag=True)
@click.option("-ibn", "--input_batch_norm", is_flag=True)
@click.option("-mt", "--max_tokens", type=int, default=None)
@click.option("--multi_label/--single_label", default=True)
@click.option("--parallel/--not-parallel", default=False)
def main(encoder,
         multi_label,
         test_frac,
         batch_size,
         epochs,
         learning_rate,
         learning_rate_finetune,
         dropout_rate,
         batch_norm,
         input_batch_norm,
         max_tokens,
         parallel):
    logger = create_logger()
    logger.info("Run PBK classification training with arguments: %s", ' '.join(sys.argv[1:]))

    spark = (
        SparkSession
        .builder
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.21.1,com.linkedin.sparktfrecord:spark-tfrecord_2.12:0.3.4")
        .appName("mtdata")
        .getOrCreate()
    )

    spark.conf.set("viewsEnabled", "true")
    spark.conf.set("temporaryGcsBucket", "temp-spark-data")

    train(
        spark, 
        logger,
        encoder,
        multi_label,
        test_frac,
        batch_size,
        epochs,
        learning_rate,
        learning_rate_finetune,
        dropout_rate,
        batch_norm,
        input_batch_norm,
        max_tokens,
        parallel)
    