import tensorflow as tf
import tensorflow.keras as nn

from mt.data.dataset.parser import ExampleParser
from mt.data.dataset.callbacks.tokenizer_callback import TokenizerCallback
from mt.data.dataset.sampler import EasyNegativeSampler
from mt.data.dataset.callbacks.feature_callbacks import FeatureTransformationCallback
from mt.models.lse.encoder import DSSM, USE, AttnDSSM
from mt.models.callbacks import VisualizeEmbeddingsCallback, SaveEncoderCallback, TransformerLRSchedule, LossHistory
from mt.models.lse.learning_algorithms import EfficientContrastiveLearner, InBatchLearner
from mt.models.losses import bpr_max, softmax_crossentropy_loss
from mt.tokenizer.tokenizer_io import load_bert_tokenizer_from_vocab_path
from mt.tokenizer.train_bert_tokenizer import main as train_bert_tokenizer
from mt.utils import download_all_tfrecords_from_s3, S3Url, get_latest_subfolder_from_url, s3_folder_exists_and_not_empty
from mt.config import config

import boto3
import os
import click

url = config.MT_OUTPUT_URL
s3url = S3Url(url)


def download_files():
    if not os.path.exists("./click"):
        download_all_tfrecords_from_s3(config.MT_DR_BANNER_URL,
                                       destination="click",
                                       timestamp_format=config.TIMESTAMP_SIGNATURE_FORMAT,
                                       tfrecord_prefix="training_data_only_clicks_mar22",
                                       preserve_subfolders=True)


def get_encoder(encoder_name, vocab, learning_rate):

    def get_use():
        d_model = 256
        encoder = USE(len(vocab), embedding_dim=d_model, dff=1024, num_attn_heads=4,
                      num_attn_layers=2, dropout_rate=0.1)
        learning_rate = TransformerLRSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        return encoder, optimizer

    def get_dssm():
        encoder = DSSM(len(vocab), 300, [300, 300],
                       batch_norm=True, dropout_rate=0.3)
        optimizer = nn.optimizers.Adam(learning_rate=learning_rate, clipnorm=3.0)
        return encoder, optimizer

    def get_dssm_w_attn():
        encoder = AttnDSSM(len(vocab), 300, 128)
        optimizer = nn.optimizers.Adam(learning_rate=learning_rate, clipnorm=3.0)
        return encoder, optimizer

    if encoder_name == "dssm":
        encoder, optimizer = get_dssm()
    elif encoder_name == "use":
        encoder, optimizer = get_use()
    elif encoder_name == "dssm_w_attn":
        encoder, optimizer = get_dssm_w_attn()

    return encoder, optimizer


@click.command()
@click.option("-e", "--encoder_name", type=str, required=True)
@click.option("-l", "--loss", type=str, required=True)
@click.option("-nn", "--num_negs", type=int, default=5)
@click.option("-bs", "--batch_size", type=int, default=128)
@click.option("-ep1", "--epochs1", type=int, default=3)
@click.option("-ep2", "--epochs2", type=int, default=5)
@click.option("-lr", "--learning_rate", type=float, default=1e-3)
@click.option("-al", "--alpha_lower", type=float, default=0.0)
@click.option("-au", "--alpha_upper", type=float, default=0.0)
@click.option("-mt", "--max_tokens", type=int, default=None)
@click.option("-s1", "--steps1", type=int, default=None)
@click.option("-s2", "--steps2", type=int, default=None)
def main(encoder_name,
         loss,
         num_negs,
         batch_size,
         epochs1,
         epochs2,
         learning_rate,
         alpha_lower,
         alpha_upper,
         max_tokens,
         steps1,
         steps2):

    loss_fn = {
        "bpr": bpr_max,
        "cross_entropy": softmax_crossentropy_loss
    }[loss]

    if steps1:
        epochs1 = 1
    if steps2:
        epochs2 = 1

    client = boto3.client("s3")
    project_folder = get_latest_subfolder_from_url(
        client, s3url, config.TIMESTAMP_SIGNATURE_FORMAT)
    s3_vocab_prefix = os.path.join(
        config.MT_OUTPUT_URL, project_folder, config.BERT_VOCAB_SUFFIX)
    s3_vocab_url = S3Url(os.path.join(
        s3_vocab_prefix, config.BERT_VOCAB_FILENAME))
    if not s3_folder_exists_and_not_empty(client, s3_vocab_prefix):
        train_bert_tokenizer("tmp/*.tfrecord", merge_vocabs=True)
    tokenizer, vocab = load_bert_tokenizer_from_vocab_path(
        s3_vocab_url, return_vocab=True)

    feature_map = {col: tf.io.FixedLenFeature(
        [], tf.string) for col in config.BERT_VOCABS}
    parser = ExampleParser(feature_spec=feature_map, on_batch=False)
    sampler = EasyNegativeSampler(num_negs)

    if not max_tokens:
        seq_length = None
    else:
        seq_length = {
            config.QUERY_COL: max_tokens,
            config.PRODUCT_TITLE_COL: max_tokens}

    token_ops = TokenizerCallback(tokenizer=tokenizer,
                                  cols=[config.QUERY_COL,
                                        config.PRODUCT_TITLE_COL],
                                  max_length=seq_length)

    cb = FeatureTransformationCallback(column_operation_mappings={
        config.QUERY_COL: lambda x: tf.strings.strip(x),
        config.PRODUCT_TITLE_COL: lambda x: tf.strings.strip(x)
    })

    def get_ds_random_samples(pattern):
        ds = (
            tf.data.Dataset.list_files(pattern, shuffle=True)
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            .shuffle(1_000_000)
            .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
            .map(cb, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(lambda x: tf.reduce_all(tf.not_equal(x["searchterm"], b'')))
            .batch(num_negs+1, drop_remainder=True)
            .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
            .map(sampler, num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
            .shuffle(10_000)
            .batch(batch_size, drop_remainder=True)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    train_ds_random_samples = get_ds_random_samples("click/train/*.tfrecord")
    val_ds_random_samples = get_ds_random_samples("click/val/*.tfrecord")

    encoder, optimizer = get_encoder(encoder_name, vocab, learning_rate)

    model_url = S3Url(os.path.join(
        config.MT_DR_BANNER_URL, project_folder, "model/"))
    # callbacks
    save_callback = SaveEncoderCallback(encoder.name, model_name=encoder.name,
                                        s3_path=model_url, save_best_only=True, save_weights_only=False, verbose=1)

    emb_viz = VisualizeEmbeddingsCallback(token_ops, num_products=500)

    lr_callback = nn.callbacks.ReduceLROnPlateau(verbose=1)

    hist_easy = LossHistory(record_every_n=100, val_ds=val_ds_random_samples.take(8))

    model1_callbacks = [hist_easy]
    if not isinstance(optimizer.learning_rate, nn.optimizers.schedules.LearningRateSchedule):
        model1_callbacks.append(lr_callback)

    model2_callbacks = [save_callback, emb_viz]
    if not isinstance(optimizer.learning_rate, nn.optimizers.schedules.LearningRateSchedule):
        model1_callbacks.append(lr_callback)

    model1 = EfficientContrastiveLearner(encoder, loss=loss_fn, normalize=True)
    model1.compile(optimizer=optimizer)
    history = model1.fit(train_ds_random_samples,
                         validation_data=val_ds_random_samples,
                         epochs=epochs1, 
                         callbacks=model1_callbacks, 
                         verbose=1,
                         steps_per_epoch=steps1,
                         validation_steps=30)

    def get_ds_hard(pattern):
        ds = (
            tf.data.Dataset.list_files(pattern, shuffle=True)
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(1_000_000)
            .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
            .map(cb, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(lambda x: tf.reduce_all(tf.not_equal(x["searchterm"], b'')))
            .batch(1_000)
            .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
            .batch(batch_size, drop_remainder=True)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    train_ds_hard = get_ds_hard("click/train/*.tfrecord")
    val_ds_hard = get_ds_hard("click/val/*.tfrecord")

    hist_hard = LossHistory(record_every_n=400, val_ds=val_ds_hard.take(8))
    model2_callbacks.append(hist_hard)

    model2 = InBatchLearner(encoder, batch_size, graph_exec=True, loss=loss_fn,
                            num_negatives=num_negs, alpha_bounds=[alpha_lower, alpha_upper])

    model2.compile(optimizer=optimizer, run_eagerly=False)
    history = model2.fit(train_ds_hard,
                         validation_data=val_ds_hard, 
                         epochs=epochs2, 
                         callbacks=model2_callbacks,
                         steps_per_epoch=steps2,
                         validation_steps=30)