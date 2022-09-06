import tensorflow as tf
import tensorflow.keras as nn
import tensorflow_ranking as tfr

from mt.config import config
from mt.models.layers import KNRMLayer, PbkClassification, SemanticMatchingScorer, create_tower, MultiTaskScoringFn
from mt.models.batch_norm import BatchNormalization

from typing import List


class MLPRank(nn.Model):

    def __init__(
            self, 
            encoder: nn.Model,
            classifier: nn.Model, 
            pbks: List[str],
            hidden_layers: List[int]=[128,128],
            batch_norm: bool = True, 
            embedding_layer=None,
            dropout_rate=0.5,
            input_batch_norm=False,
            num_tasks=1,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        if embedding_layer:
            self.knrm = KNRMLayer(embedding_layer, sequence_dim=1, trainable=False)  # training is slow
        self.semantic_scorer = SemanticMatchingScorer(encoder)
        self.pbk_matcher = PbkClassification(classifier, pbks)

        if input_batch_norm:
            self.input_batch_norm = BatchNormalization(mask_value=-1.0)

        if num_tasks > 1:
            self.scoring_function = MultiTaskScoringFn(num_tasks,
                                                       expert_units=hidden_layers[0],
                                                       tower_units=hidden_layers,
                                                       dropout=dropout_rate)
        else:
            self.scoring_function = create_tower(hidden_layers,
                                    output_units=1, 
                                    activation="relu", 
                                    output_activation=None, 
                                    input_batch_norm=False, 
                                    use_batch_norm=batch_norm, 
                                    dropout=dropout_rate)

            self.scoring_function.add(nn.layers.Lambda(lambda x: tf.squeeze(x, -1)))
 
    def call(self, inputs, training=True):
        # [B, PN]
        mask = tf.not_equal(tf.reduce_sum(inputs[config.PRODUCT_TITLE_COL], axis=-1), 0)
        # [B, PN, M]
        num_features = inputs[config.NUMERIC_FEATURES_COL]
        sms = self.semantic_scorer(inputs)
        pbk_match = self.pbk_matcher(inputs)

        # [B, PN, K+M]
        x = tf.concat([num_features, sms, pbk_match], axis=-1)

        if hasattr(self, "knrm"):
            # [B, PN, Q, K]
            kernel_features = self.knrm(inputs, training)
            # [B, PN, K]
            kernel_features = tf.reduce_sum(kernel_features, axis=2)
            kernel_features = tf.where(mask[...,tf.newaxis], kernel_features, -1.0)
            x = tf.concat([x, kernel_features], axis=-1)

        if hasattr(self, "input_batch_norm"):
            x = self.input_batch_norm(x, training=training)

        # mask padded items to feature values of zero -> no influence, no gradients
        x = tf.where(mask[...,tf.newaxis], x, 0.0)

        score = self.scoring_function(x, training=training)
        # score = tf.squeeze(score, axis=-1)

        return score