import tensorflow as tf
import tensorflow.keras as nn

from mt.models.layers import AttentionLayer, PbkClassification, MatchHistogram, create_tower
from mt.config import config


class DRMM(nn.Model):

    def __init__(
            self, 
            encoder: nn.Model, 
            nodes: list = [1],
            inner_activations: str = None,
            output_activation: str = "sigmoid",
            dropout_rate:float = None,
            batch_norm:bool = False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder #nn.layers.Embedding(config.VOCAB_SIZE, 300) #model
        self.mlp = create_tower(hidden_layer_dims=nodes[:-1],
                                output_units=nodes[-1],
                                activation=inner_activations,
                                output_activation=output_activation,
                                use_batch_norm=batch_norm,
                                dropout=dropout_rate)
        
        # MultiLayerPerceptron(nodes, inner_activations, output_activation, dropout_rate, batch_norm)
        self.att_layer = AttentionLayer()

    def call(self, inputs, training, return_attention=False):
        """
        B = batch size
        L = query sequence length
        PN = number of examples per instance (positives + negatives)
        BS = bin size
        E = embedding size
        """
        # [B, L]
        query = inputs[config.QUERY_COL]
        batch_size = tf.shape(query)[0]      
        anchor_seq_len = tf.shape(query)[1]
        
        histogram = inputs[config.MATCH_HIST_COL]
        bin_size = tf.shape(histogram)[-1]
        # [B, PN, L, BS]
        histogram = tf.reshape(histogram, (batch_size, -1, anchor_seq_len, bin_size))
        
        # [BS, L, E]
        embed_query = self.encoder(query)

        # shape = [B, L]
        atten_mask = tf.not_equal(query, 0)
        # shape = [B, L]
        atten_mask = tf.cast(atten_mask, tf.float32)
        # shape = [B, L, E]
        atten_mask = tf.expand_dims(atten_mask, axis=2)
        # shape = [B, L, 1]
        attention_probs = self.att_layer(embed_query, atten_mask)
        
        # shape = [B, PN, L, 1]
        histogram_to_dense_output = self.mlp(histogram, training=training)
         # [B, L, 1] -> [B, 1, L, 1] * [B, PN, L, 1] -> [B, PN, L, 1] -> [B, PN, 1]
        score = tf.reduce_sum(attention_probs[:, tf.newaxis, ...] * histogram_to_dense_output, axis=2)
        
        # [BS, PN, 1] -> [BS, PN]
        score = tf.squeeze(score, axis=-1)
        if return_attention:
            return score, attention_probs
        return score


class DHRMM(nn.Model):
    """deep hybrid relevance matching model"""

    def __init__(
            self, 
            encoder: nn.layers.Layer, 
            pbk_classifier: nn.Model,
            pbks: list,
            bin_size: int = 30, 
            hidden_layers: list = [512, 512],
            inner_activations: str = "relu",
            output_activation: str = None,
            dropout_rate:float = None,
            batch_norm:bool = True,
            input_batch_norm: bool = True, 
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.encoder.trainable = False
        self.bin_size = bin_size
        self.histogram_layer = MatchHistogram(self.encoder, bin_size=self.bin_size, trainable=False)
        self.pbk_matcher = PbkClassification(pbk_classifier, pbks)

        self.mlp = create_tower(hidden_layer_dims=hidden_layers,
                                output_units=1,
                                activation=inner_activations,
                                output_activation=output_activation,
                                input_batch_norm=input_batch_norm,
                                use_batch_norm=batch_norm,
                                dropout=dropout_rate)

        self.num_feat_mlp = create_tower(hidden_layer_dims=hidden_layers,
                                         output_units=1,
                                         activation=inner_activations,
                                         output_activation=output_activation,
                                         input_batch_norm=input_batch_norm,
                                         use_batch_norm=batch_norm,
                                         dropout=dropout_rate)

        self.att_layer = AttentionLayer()
        # self.output_activation = nn.layers.Activation(norm_output_with)


    def call(self, inputs, training, return_attention=False):
        """
        B = batch size
        L = query sequence length
        PN = number of examples per instance (positives + negatives)
        BS = bin size
        E = embedding size
        M = dimension of the numerical feature space
        """
        # [B, PN]
        # mask = tf.not_equal(inputs[config.CLICK_COL], -1)
        # [B, L]
        query = inputs[config.QUERY_COL]
        batch_size = tf.shape(query)[0]      
        anchor_seq_len = tf.shape(query)[1]
        # [B, L+1] # add special token [1]
        query_w_cls = tf.concat([tf.ones((batch_size, 1), dtype=tf.int64), query], axis=1)
        # 
        histogram = tf.stop_gradient(self.histogram_layer(inputs))
        # [B, PN, L, BS]
        histogram = tf.reshape(histogram, (batch_size, -1, anchor_seq_len, self.bin_size))
        
        num_features = inputs[config.NUMERIC_FEATURES_COL]
        pbk_match = self.pbk_matcher(inputs)
        num_features = tf.concat([num_features, pbk_match], axis=-1)
        feat_dim = tf.shape(num_features)[-1]

        # [B, PN, M]
        num_features = tf.reshape(num_features, (batch_size, -1, feat_dim))
        # [B, PN, 1, 1]
        num_features_score = self.num_feat_mlp(num_features[:, :, tf.newaxis, :], training=training)

        # [BS, L+1, E]
        embed_query = self.encoder(query_w_cls, training=training)

        # shape = [B, L+1]
        atten_mask = tf.not_equal(query_w_cls, 0)
        # shape = [B, L+1]
        atten_mask = tf.cast(atten_mask, tf.float32)
        # shape = [B, L+1, E]
        atten_mask = tf.expand_dims(atten_mask, axis=2)
        # shape = [B, L+1, 1]
        attention_probs = self.att_layer(embed_query, atten_mask)
        
        # shape = [B, PN, L, 1]
        histogram_to_dense_output = self.mlp(histogram, training=training)
        # shape = [B, PN, L+1, 1]
        raw_and_num_dense_output = tf.concat([num_features_score, histogram_to_dense_output], axis=2)
         # [B, L+1, 1] -> [B, 1, L+1, 1] * [B, PN, L+1, 1] -> [B, PN, L+1, 1] -> [B, PN, 1]
        score = tf.reduce_sum(attention_probs[:, tf.newaxis, ...] * raw_and_num_dense_output, axis=2)
        
        # [BS, PN, 1] -> [BS, PN]
        score = tf.squeeze(score, axis=-1)
        # mask score for softmax
        # score += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        # score = self.output_activation(score)
        if return_attention:
            return score, attention_probs
        return score
