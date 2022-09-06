# simple scoring function 
import tensorflow as tf
import tensorflow.keras as nn
import tensorflow_ranking as tfr

from mt.config import config
from mt.models.layers import create_tower


class SimScorer(nn.Model):

    def __init__(self, layer_units=[128, 128], activation="relu", output_activation=None, bn=True, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scoring_function = create_tower(layer_units,
                                             output_units=1,
                                             activation=activation,
                                             output_activation=output_activation,
                                             input_batch_norm=False,
                                             use_batch_norm=bn,
                                             dropout=dropout)

        self.scoring_function.add(nn.layers.Lambda(lambda x: tf.squeeze(x, -1)))
 
    def call(self, inputs, training=True):

        # [B, PN, M]
        x = inputs[config.NUMERIC_FEATURES_COL]

        score = self.scoring_function(x, training=training)
        # score = tf.squeeze(score, axis=-1)

        return score