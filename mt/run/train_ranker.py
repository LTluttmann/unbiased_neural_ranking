from mt.models.model_io import s3_get_keras_model
from mt.tokenizer.tokenizer_io import load_bert_tokenizer_from_vocab_path, read_vocab
from mt.utils import S3Url, download_all_tfrecords_from_s3

from mt.models.ltr.attnrank import AttnRank
from mt.models.ultr.joe import JointEstimator
from mt.models.ltr import espec, espec_val, cspec
from mt.models.callbacks import SaveRankerCallback, TransformerLRSchedule
from mt.models.lse.pbk import PBKClassifier
from mt.models.model_io import s3_get_keras_model_from_weights

from mt.data.dataset.parser import SeqExampleParser as SeqExampleParser
from mt.data.dataset.callbacks.tokenizer_callback import TokenizerCallback
from mt.data.dataset.sampler import SequenceExampleSampler
from mt.data.dataset.parser import ExampleParser
from mt.data.dataset.callbacks.feature_callbacks import FeatureTransformationCallback, FeatureMerger, LayoutEncoder, NormalizeAlongAxis
from mt.config import config

import boto3
import tensorflow as tf
import tensorflow_ranking as tfr
import os

ADDITIONAL_RANDOM_NEGATIVES = 5
NUM_NEGATIVES = 5
BATCH_SIZE = 128


@tf.function
def concat_from_zipped_datasets(a,b):
    """concatenates the second dimensions from two datasets"""
    a = a.copy()
    query = a.pop(config.QUERY_COL)
    batch_size = tf.shape(query)[0]

    for k,v in a.items():
        if k in config.QUERY_ITEM_FEATURES + config.POSITION_BIAS_FEATURES + [config.CLICK_COL, config.ORDER_COL]:
            random_negatives = tf.zeros_like(tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0))
        elif k == "sampling_weights":
            random_negatives = tf.ones_like(tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0))
        else:
            random_negatives = tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0)
        a[k] = tf.concat((v, random_negatives), axis=1)
    a[config.QUERY_COL] = query
    return a


def get_pbk_classifier(encoder, tokenizer, pbks, client):

    pbk_classifier = PBKClassifier(encoder, pbks)

    if not config.MAX_TOKENS:
        seq_length=None
    else:
        seq_length={
            "query": config.MAX_TOKENS
        }
    token_ops = TokenizerCallback(
        tokenizer=tokenizer, 
        cols=["query"],
        max_length=seq_length
    )

    q = "nintendo switch"
    q = tokenizer.tokenize(q).merge_dims(-2,-1)
    if config.MAX_TOKENS:
        q = token_ops.pad_sequence_new(q, config.MAX_TOKENS, [1, config.MAX_TOKENS])

    pbk_classifier(q)
    classifier_url = S3Url(os.path.join(config.MT_DR_BANNER_URL, "20220404-14:40:51/model/pbk_classifier.h5"))
    s3_get_keras_model_from_weights(pbk_classifier, classifier_url.bucket, classifier_url.key, client=client)
    return pbk_classifier


def download_datasets(seq_ds, click_ds):
    if not os.path.exists("seq_data"):
        download_all_tfrecords_from_s3(config.MT_DR_BANNER_URL, destination="seq_data", timestamp_format=config.TIMESTAMP_SIGNATURE_FORMAT, tfrecord_prefix=seq_ds, preserve_subfolders=True)
    if not os.path.exists("click"):
        download_all_tfrecords_from_s3(config.MT_DR_BANNER_URL, destination="click", timestamp_format=config.TIMESTAMP_SIGNATURE_FORMAT, tfrecord_prefix=click_ds, preserve_subfolders=True)



def add_sample_weight(x):
    w = tf.maximum(1.0, tf.cast(x[config.ORDER_COL], tf.float32) * tf.squeeze(x[config.PRICE_COL], -1))
    x[config.SAMPLE_WEIGHT_ON_LOSS_COL] = w
    return x


def main(seq_ds, click_ds):

    download_datasets(seq_ds, click_ds)

    vocab_url = S3Url("XXXXXXXXX")
    tokenizer, vocab = load_bert_tokenizer_from_vocab_path(vocab_url, return_vocab=True)#, basic_tokenizer_class=AccentPreservingBasicTokenizer, lower_case=True)

    pbk_url = S3Url("XXXXXXXX")
    client = boto3.client("s3")
    client.download_file(pbk_url.bucket, pbk_url.key, "pbks.txt")
    pbks = read_vocab("pbks.txt")

    pbk_lookup = tf.keras.layers.StringLookup(vocabulary=pbks)
    invert_lookup = tf.keras.layers.StringLookup(vocabulary=pbks, invert=True)

    encoder_url = S3Url(os.path.join(config.MT_DR_BANNER_URL, "/20220404-14:40:51/model/"))
    encoder = s3_get_keras_model("attn_dssm_4", encoder_url)

    pbk_classifier = get_pbk_classifier(encoder, tokenizer, pbks, client)

    token_ops = TokenizerCallback(
        tokenizer=tokenizer, 
        cols=[config.QUERY_COL, config.PRODUCT_TITLE_COL],
        max_length={
            config.QUERY_COL: config.MAX_TITLE_LENGTH, # TODO !!!!! ENCODER NEEDS SAME
            config.PRODUCT_TITLE_COL: config.MAX_TITLE_LENGTH}
    )

    def pbk_match(x):
        q = x[config.QUERY_COL]
        class_pred = pbk_classifier(q, training=False)
        class_pred = class_pred[:, tf.newaxis, :]
        pbk_one_hot = tf.one_hot(pbk_lookup(x[config.PBK_COL]), depth=len(pbks)+1)
        class_feature = tf.reduce_sum(pbk_one_hot * class_pred, axis=-1)
        x["pbk_match"] = class_feature
        return x

    def calc_semantic_matching_score(x):
        bs = tf.shape(x[config.PRODUCT_TITLE_COL])[0]
        seq_len = tf.shape(x[config.PRODUCT_TITLE_COL])[1]
        token_len = tf.shape(x[config.PRODUCT_TITLE_COL])[2]
        
        q_emb = encoder(x[config.QUERY_COL], training=False)
        emb_dim = tf.shape(q_emb)[1]
        
        doc_emb = encoder(tf.reshape(x[config.PRODUCT_TITLE_COL], (-1, token_len)), training=False)
        doc_emb = tf.reshape(doc_emb, (bs, seq_len, emb_dim))
        sms = tf.keras.layers.Dot(axes=[1,2], normalize=True)([q_emb, doc_emb])
        sms = tf.where(tf.math.is_nan(sms), tf.zeros_like(sms), sms)
        x["sms"] = sms
        return x

    feature_transformations = {k: lambda x: tf.expand_dims(tf.math.log1p(tf.maximum(0.0, tf.cast(x, tf.float32))), -1) for k in config.LOG1P_TRANSFORM_COLS}

    for feat in config.NUMERICAL_COLUMNS:
        feature_transformations[feat] = feature_transformations.get(feat, lambda x: tf.expand_dims(x, -1))

    transform_callback = FeatureTransformationCallback(
        column_operation_mapping=feature_transformations)

    normalize_num_features = NormalizeAlongAxis(column=config.NUMERIC_FEATURES_COL, axis=1, mask_value=-1, kind="z_norm")
    normalize_judgements = NormalizeAlongAxis(column=config.JUDGEMENT_COL, axis=1, mask_value=-1, kind="min_max")

    device_lookup = tf.keras.layers.StringLookup(vocabulary=config.DEVICE_VOCAB)
    layout_lookup = tf.keras.layers.StringLookup(vocabulary=config.LAYOUT_VOCAB)

    layout_kwargs = {
        "interaction_cols":None, 
        "merge_cols": config.POSITION_BIAS_FEATURES, 
        "column_operation_mapping":{
            # position starts with one, hence subtract by one, since one_hot_expects values starting from 0
            config.POSITION_COL: lambda x: tf.one_hot(tf.cast(x-1, tf.int64), config.MAX_SEQ_LENGTH),
            # lookup layer starts with one and assigns unknows a zero. We want zeros to be masked out, hence
            # let the lookup values start from 0. Unkowns become -1, which one_hot maps to vector of all zeros.
            config.DEVICE_COL: lambda x: tf.one_hot(device_lookup(x)-1, len(config.DEVICE_VOCAB)),
            config.LAYOUT_COL: lambda x: tf.one_hot(layout_lookup(x)-1, len(config.LAYOUT_VOCAB))
        },
        "feature_name": config.POS_BIAS_FEATURE_COL
    }
    layout_encoder = LayoutEncoder(**layout_kwargs)

    click_spec = espec.copy()

    ex_par = ExampleParser(feature_spec=click_spec)
    seq_par = SeqExampleParser(feature_spec=espec, context_feature_spec=cspec, list_size=200)
    seq_par_val = SeqExampleParser(feature_spec=espec_val, context_feature_spec=cspec, list_size=config.MAX_SEQ_LENGTH)

    #sampler = SequenceExampleSampler(num_negatives=11, replacement=False, sample_weight="sampling_weights")
    sampler = SequenceExampleSampler(num_negatives=NUM_NEGATIVES, replacement=False, sample_weight=config.SAMPLING_WEIGHT_COL)

    num_merge_callback = FeatureMerger(config.NUMERICAL_COLUMNS, merged_feature_name=config.NUMERIC_FEATURES_COL)

    ds_val = (
        tf.data.Dataset.list_files("seq_data/val/*.tfrecord")
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10_000)
        .batch(64)
        .map(seq_par_val, num_parallel_calls=tf.data.AUTOTUNE)
        .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
        .map(calc_semantic_matching_score, num_parallel_calls=tf.data.AUTOTUNE)
        .map(pbk_match, num_parallel_calls=tf.data.AUTOTUNE) 
        .map(transform_callback, num_parallel_calls=tf.data.AUTOTUNE)
        .map(num_merge_callback, num_parallel_calls=tf.data.AUTOTUNE)
        .map(normalize_num_features, num_parallel_calls=tf.data.AUTOTUNE)
        .map(normalize_judgements, num_parallel_calls=tf.data.AUTOTUNE)
        #.map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
        .map(layout_encoder, num_parallel_calls=tf.data.AUTOTUNE)
        .take(8)
        .cache()
    )

    ds_negs = (
        tf.data.Dataset.list_files("click/train/*.tfrecord", shuffle=True)
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1_000_000)
        .map(ex_par, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(ADDITIONAL_RANDOM_NEGATIVES)
    )

    ds = (
        tf.data.Dataset.list_files("seq_data/train/*.tfrecord", shuffle=True)
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(100_000)
        .batch(128)
        .map(seq_par, num_parallel_calls=tf.data.AUTOTUNE)
    )

    ds_zip = (
        tf.data.Dataset.zip((ds, ds_negs))
        .map(concat_from_zipped_datasets, num_parallel_calls=tf.data.AUTOTUNE)
        
        .map(sampler, num_parallel_calls=tf.data.AUTOTUNE)
        
        .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
        .map(calc_semantic_matching_score, num_parallel_calls=tf.data.AUTOTUNE)
        .map(pbk_match, num_parallel_calls=tf.data.AUTOTUNE) 
        
        .map(transform_callback, num_parallel_calls=tf.data.AUTOTUNE)
        .map(num_merge_callback, num_parallel_calls=tf.data.AUTOTUNE)
        .map(normalize_num_features, num_parallel_calls=tf.data.AUTOTUNE)
        
        # .map(token_ops, num_parallel_calls=tf.data.AUTOTUNE)
        .map(layout_encoder, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    attnrank = AttnRank(
        encoder, 
        #classifier=pbk_classifier,
        #pbks=pbks,
        num_attn_heads=8, 
        attn_head_size=512, 
        num_attn_layers=4, 
        hidden_layers=[512, 512])

    
    save_ranker = SaveRankerCallback("attnrank", model_name=attnrank.name, s3_path=encoder_url, verbose=1)
    lr = TransformerLRSchedule(512)

    attnrank_trainer = JointEstimator(
        ranker=attnrank, 
        num_negatives=11, # set -1 for bpr max loss
        train_end_to_end=True, 
        learning_rate = lr,
        joe_nodes=[128, 1],
        joe_activations="tanh",
        joe_output_activation="linear",
        joe_multiplicative=False,
        temperature=1.0,
        joe_dropout=0.3,
        metrics={
            ("hierarchical_click_judgement", "relevance_scores"): tfr.keras.metrics.NDCGMetric(),
            (config.CLICK_COL, "relevance_scores"): tfr.keras.metrics.MRRMetric()}
    )

    attnrank_trainer.fit(ds_zip, validation_data=ds_val, epochs=2, callbacks=[save_ranker])