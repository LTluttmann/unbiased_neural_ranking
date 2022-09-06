import tensorflow as tf
from mt.config import config
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

espec = {
    config.PRODUCT_TITLE_COL: tf.io.FixedLenFeature([], tf.string, default_value=""),
    config.CLICK_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.ORDER_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.POSITION_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.LAYOUT_COL: tf.io.FixedLenSequenceFeature([], tf.string),
    config.DEVICE_COL: tf.io.FixedLenSequenceFeature([], tf.string),
    config.RATING_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.NUM_RATINGS_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.DISCOUNT_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.PRICE_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.NUM_CLICKS_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.BRAND_MATCH_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.TFIDF_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.BM25_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.DEMAND_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.OFFER_OR_PRODUCT_COL: tf.io.FixedLenSequenceFeature([], tf.string),
    config.PBK_COL: tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    config.SAMPLING_WEIGHT_COL: tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0), # default zero for zero sampling probability (see sampler)
    config.TARGET_MATCH_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.COLOR_MATCH_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.AVAILABILITY_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.FRESHNESS_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.RETRIEVED_PRODUCTS_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.PAGE_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.QUERY_LENGTH_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.QUERY_FAME_COL: tf.io.FixedLenSequenceFeature([], tf.int64),
    config.SHIPPING_COST_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.COEC_COL: tf.io.FixedLenSequenceFeature([], tf.float32),
    config.CATEGORY_COL: tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
}
espec = {k:v for k,v in espec.items() if k in config.ALL_COLS}

espec_val = {**espec, config.JUDGEMENT_COL: tf.io.FixedLenSequenceFeature([], tf.float32, default_value=-1)}

cspec = {
    config.QUERY_COL: tf.io.FixedLenFeature([1], tf.string),
}

