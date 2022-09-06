import tensorflow as tf
from mt.config import config
    
feature_map = {col: tf.io.FixedLenFeature([], tf.string) for col in config.BERT_VOCABS}