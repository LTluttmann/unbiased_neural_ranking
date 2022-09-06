import tensorflow as tf
import tensorflow.keras as nn

from mt.models.layers import AttentionLayer, KNRMLayer, create_tower, PbkClassification
from mt.config import config



class HybridKNRM(nn.Model):

    def __init__(self,
                 encoder: nn.Model,
                 pbk_classifier: nn.Model,
                 pbks: list,
                 kernel_num: int = 11,
                 sigma=0.1,
                 exact_sigma=0.001,
                 nodes: list = [1],
                 inner_activations: str = None,
                 output_activation: str = "sigmoid",
                 dropout_rate: float = None,
                 batch_norm: bool = False,
                 input_batch_norm: bool = True,
                 *args, **kwargs):
                 
        super().__init__(*args, **kwargs)
        self.encoder = encoder #nn.layers.Embedding(config.VOCAB_SIZE, 300) #model
        self.mlp = create_tower(hidden_layer_dims=nodes,
                                output_units=1,
                                activation=inner_activations,
                                output_activation=output_activation,
                                use_batch_norm=batch_norm,
                                input_batch_norm=input_batch_norm,
                                dropout=dropout_rate,
                                mask_value=-1.0)

        self.num_feat_mlp = create_tower(hidden_layer_dims=nodes[:-1],
                                         output_units=1,
                                         activation=inner_activations,
                                         output_activation=output_activation,
                                         input_batch_norm=input_batch_norm,
                                         use_batch_norm=batch_norm,
                                         dropout=dropout_rate,
                                         mask_value=-1.0)

        self.knrm_layer = KNRMLayer(encoder, kernel_num, sigma, exact_sigma, sequence_dim=1)
        self.att_layer = AttentionLayer()
        self.pbk_matcher = PbkClassification(pbk_classifier, pbks)


    def call(self, inputs, training):
        """
        B = batch size
        Q = query length
        D = document length
        PN = number of examples per instance (positives + negatives)
        BS = bin size
        E = embedding size
        K = kernel_num
        """
        # [B, Q]
        query = inputs[config.QUERY_COL]
        # [B, PN, D]
        documents = inputs[config.PRODUCT_TITLE_COL]
        batch_size = tf.shape(query)[0]
        # [B, Q+1] # add special token [1]
        query_w_cls = tf.concat([tf.ones((batch_size, 1), dtype=tf.int64), query], axis=1)
        num_features = inputs[config.NUMERIC_FEATURES_COL]
        
        # [B, Q+1]
        atten_mask = tf.cast(tf.not_equal(query_w_cls, 0), tf.float32)
        # [B, Q+1, E]
        atten_mask = tf.expand_dims(atten_mask, axis=2)

        # [BS, Q+1, E]
        q_w_cls_embed = self.encoder(query_w_cls)

        # [B,PN,Q,K]
        kde = self.kernel_pooling(inputs, training)
        # [B,PN,Q,1]
        kde_scores = self.mlp(kde, training=training)

        # [B, PN, M] -> [B, PN, 1]
        num_features_score = self.num_feat_mlp(num_features, training=training)
        # [B, PN, Q+1, 1]
        raw_and_num_dense_output = tf.concat([num_features_score[...,tf.newaxis], kde_scores], axis=2)
        # [B, Q+1, 1]
        attention_probs = self.att_layer(q_w_cls_embed, atten_mask)
        # [B, 1, Q+1, 1] * [B,PN,Q+1,1] -> reduce_sum: [B,PN,1]
        score = tf.reduce_sum(attention_probs[:, tf.newaxis,...] * raw_and_num_dense_output, axis=2)
        # [BS, PN, 1] -> [BS, PN]
        score = tf.squeeze(score, axis=-1)

        return score
