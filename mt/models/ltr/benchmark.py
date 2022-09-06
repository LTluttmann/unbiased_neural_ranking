import tensorflow as tf
from mt.config import config


class BenchmarkScorer(tf.keras.Model):
    """this is a very simple model which shall mimic the current production ranker
    It calculates for q,d-pair in the provided dataset a score which is 1/sqrt(rank),
    where rank is the rank of the respective d under q with given production ranker"""
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        mask = tf.not_equal(inputs[config.POSITION_COL], -1)
        pos = inputs[config.POSITION_COL]
        pos = tf.cast(pos, tf.float32)
        score = tf.math.rsqrt(pos)
        score = tf.where(mask, score, pos)
        return score