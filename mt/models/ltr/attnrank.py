import tensorflow as tf
import tensorflow.keras as nn
import tensorflow_ranking as tfr

from mt.config import config
from mt.models.layers import KNRMLayer, PbkClassification, SemanticMatchingScorer, create_tower, MultiTaskScoringFn, MultiLayerPerceptron
from mt.models.batch_norm import BatchNormalization

from typing import List


INPUT_SIGNATURE = {config.NUMERIC_FEATURES_COL: nn.Input(shape=[None, len(config.NUMERICAL_COLUMNS) + sum(len(v) for v in config.CATEGORICAL_FEATURES.values())], dtype=tf.float32), 
                   config.QUERY_COL: nn.Input(shape=[config.MAX_TOKENS], dtype=tf.int64), 
                   config.PRODUCT_TITLE_COL: nn.Input(shape=[None, config.MAX_TOKENS], dtype=tf.int64),
                   config.PBK_COL: nn.Input(shape=[None, ], dtype=tf.string)}


class AttnRank(nn.Model):

    def __init__(
            self, 
            encoder: nn.Model,
            classifier: nn.Model, 
            pbks: List[str],
            num_attn_heads: int, 
            attn_head_size: int,
            num_attn_layers: int, 
            hidden_layers: List[int]=[512,512],
            embedding_layer=None,
            dropout_rate=0.5,
            dropout_mha=0.1,
            input_batch_norm=False,
            activation="relu",
            num_tasks=1,
            scoring_fn_layers: List[int]=[512],
            *args, **kwargs):

        super().__init__(*args, **kwargs)   
        self.input_signature = INPUT_SIGNATURE                                                 
        self.num_tasks = num_tasks
        
        if embedding_layer:
            self.knrm = KNRMLayer(embedding_layer, sequence_dim=1, trainable=False)  # training is slow
        self.semantic_scorer = SemanticMatchingScorer(encoder)
        self.pbk_matcher = PbkClassification(classifier, pbks)

        if input_batch_norm:
            self.input_batch_norm = BatchNormalization(mask_value=-1.0)

        self.mhsa = tfr.keras.layers.DocumentInteractionAttention(num_heads=num_attn_heads,
                                                                  head_size=attn_head_size, 
                                                                  num_layers=num_attn_layers, 
                                                                  dropout=dropout_mha)

        self.batch_norm = nn.layers.BatchNormalization()

        self.mlp = create_tower(hidden_layers,
                                output_units=hidden_layers[-1], 
                                activation=activation, 
                                dropout=dropout_rate, 
                                use_batch_norm=True)

        self.dense_context = nn.layers.Dense(hidden_layers[-1])

        self.activation = nn.layers.Activation(activation)

        if num_tasks > 1:
            self.scoring_function = MultiTaskScoringFn(num_tasks,
                                                       expert_units=scoring_fn_layers[0],
                                                       tower_units=scoring_fn_layers)
        else:
            self.scoring_function = nn.Sequential(
                [nn.layers.Dense(i, activation=activation) for i in scoring_fn_layers] +
                [nn.layers.Dense(1),
                 nn.layers.Lambda(lambda x: tf.squeeze(x, -1))]
            )


    def call(self, inputs, training=False):
        # [B, PN, M]
        num_features = inputs[config.NUMERIC_FEATURES_COL]
        sms = self.semantic_scorer(inputs)
        pbk_match = self.pbk_matcher(inputs)

        # [B, PN, K+M]
        x = tf.concat([num_features, sms, pbk_match], axis=-1)
        
        # [B, PN]
        mask = tf.not_equal(tf.reduce_sum(inputs[config.PRODUCT_TITLE_COL], axis=-1), 0)

        if hasattr(self, "knrm"):
            # [B, PN, Q, K]
            kernel_features = self.knrm(inputs, training)
            # [B, PN, K]
            kernel_features = tf.reduce_sum(kernel_features, axis=2)
            kernel_features = tf.where(mask[...,tf.newaxis], kernel_features, -1.0)
            x = tf.concat([x, kernel_features], axis=-1)

        # x = nn.layers.Masking(mask_value=-1)(x) # sets all masked values to zero

        if hasattr(self, "input_batch_norm"):
            x = self.input_batch_norm(x, training=training)

        # mask padded items to feature values of zero -> no influence, no gradients
        x = tf.where(mask[...,tf.newaxis], x, 0.0)

        cntxt = self.mhsa(inputs=(x, mask), training=training)
        cntxt = self.dense_context(cntxt)

        x = self.mlp(x, training=training)

        # latent cross
        out = tf.math.multiply(x, cntxt)
        out = tf.math.add(out, x)

        out = self.batch_norm(out, training=training)
        out = self.activation(out)

        scores = self.scoring_function(out, training=training)

        return scores

    def build_graph(self):
        inputs = {config.NUMERIC_FEATURES_COL: nn.Input(shape=[None, len(config.NUMERICAL_COLUMNS) + sum(len(v) for v in config.CATEGORICAL_FEATURES.values())], dtype=tf.float32), 
                   config.QUERY_COL: nn.Input(shape=[config.MAX_TOKENS], dtype=tf.int64), 
                   config.PRODUCT_TITLE_COL: nn.Input(shape=[None, config.MAX_TOKENS], dtype=tf.int64),
                   config.PBK_COL: nn.Input(shape=[None, ], dtype=tf.string)}
        return nn.Model(inputs=inputs, outputs=self.call(inputs), name=self.name)

    @property
    def name(self):
        name = super().name
        multitask = "_multitask" if self.num_tasks > 1 else ""
        return name + multitask