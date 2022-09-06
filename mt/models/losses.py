import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow.keras as nn

from mt.config import config


def binary_crossentropy_loss(from_logits=True):
    def loss_fn(y_true, y_pred, sample_weight=None):
        if y_true is None:
            raise ValueError("Need to provide a label in pointwise mode. Got None")
        loss = nn.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=from_logits)
        weighted_loss = nn.__internal__.losses.compute_weighted_loss(
            loss, sample_weight, reduction=nn.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return weighted_loss
    return loss_fn


def softmax_crossentropy_loss(temperature=None):
    temperature = temperature if temperature is not None else 1.0
    def loss_fn(_, y_pred, sample_weight=None):
        """implementation of cross entropy without label"""
        logits = y_pred / temperature

        pos_example_logits = tf.gather(logits, 0, axis=1)
        nom = tf.exp(pos_example_logits)
        # logits: [BS, (P+N)] -> [BS]
        denom = tf.reduce_sum(tf.exp(logits), axis=1)
        # [BS]
        loss = - tf.math.log(nom / denom)
        weighted_loss = nn.__internal__.losses.compute_weighted_loss(
            loss, sample_weight, reduction=nn.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return weighted_loss
    return loss_fn

def softmax_crossentropy_loss_new(temperature=None):
    temperature = temperature if temperature is not None else 1.0
    def loss_fn(y_true, y_pred, sample_weight=None):
        """implementation of cross entropy without label"""
        # replace mask values with zeros, otherwise loss will be weird. Masked values
        # are weighted with zero anyways.
        logits = y_pred / temperature
        y_true = tf.maximum(tf.cast(0, tf.int64), y_true)
        loss = nn.losses.binary_crossentropy(y_true, logits, from_logits=True)
        weighted_loss = nn.__internal__.losses.compute_weighted_loss(
            loss, sample_weight, reduction=nn.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return weighted_loss
    return loss_fn


def sigmoid_crossentropy_loss():
    def loss_fn(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.maximum(tf.cast(0, tf.int64), y_true), tf.float32)
        loss = tfr.keras.losses.SigmoidCrossEntropyLoss()
        return loss(y_true, y_pred)
    return loss_fn

def bpr_max():
    def loss_fn(_, y_pred, sample_weight=None):
        pos_example_logits = y_pred[:, 0:1, ...]
        neg_example_logits = y_pred[:, 1:, ...]
        softmax_scores = tf.nn.softmax(neg_example_logits)
        diff = pos_example_logits - neg_example_logits
        sigmoid = tf.nn.sigmoid(diff)
        loss = -tf.math.log(tf.reduce_sum(sigmoid*softmax_scores, axis=1) + 1e-24)
        weighted_loss = nn.__internal__.losses.compute_weighted_loss(
            loss, sample_weight, reduction=nn.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return weighted_loss
    return loss_fn


def triplet_loss(margin:float=1.0):
    def loss_fn(y_true, y_pred, sample_weight=None):
        D_ap = 1.0 - tf.gather(y_pred, tf.where(tf.equal(y_true, 1)), batch_dims=1)
        D_an = 1.0 - tf.gather(y_pred, tf.where(tf.equal(y_true, 0)), batch_dims=1)
        loss =  tf.maximum(0.0, margin + D_ap - D_an)
        weighted_loss = nn.__internal__.losses.compute_weighted_loss(
            loss, sample_weight, reduction=nn.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return weighted_loss
    return loss_fn