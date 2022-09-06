import tensorflow as tf
import tensorflow.keras as nn

from mt.models.layers import create_tower, EncoderLayer, AttentionLayer, KNRMLayer
from mt.config import config

from typing import List
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class DSSM(nn.Model):

    def __init__(
        self, 
        vocab_size:int,
        embedding_dim: int,
        dense_layer_nodes:List[int] = None, 
        activation:str = "relu", 
        batch_norm:bool = True,
        dropout_rate:float = None
    ):
        super().__init__()
        
        self.embedding_layer = nn.layers.Embedding(vocab_size, embedding_dim)
        # self.pool = nn.layers.GlobalAvgPool1D()
        self.masked_pooling = nn.layers.Lambda(lambda x, mask: self.pooling_w_mask(x, mask))
        if dense_layer_nodes:
            self.mlp = create_tower(hidden_layer_dims=dense_layer_nodes[:-1],
                                    output_units=dense_layer_nodes[-1],
                                    activation=activation,
                                    output_activation=None,
                                    input_batch_norm=batch_norm,
                                    use_batch_norm=batch_norm,
                                    dropout=dropout_rate)

    def pooling_w_mask(self, x, mask=None):
        if mask is not None:
            # mask (batch, tokens)
            mask = tf.cast(mask, tf.float32)
            # mask (batch, tokens, 1)
            mask = tf.expand_dims(mask, -1)
            # x (batch, tokens, edim) * mask (batch, tokens, 1)
            x = x * mask
        # (batch, edim)
        x = tf.reduce_sum(x, axis=1) / tf.reduce_sum(mask, axis=1)
        return x

    def call(self, inputs, training):
        # (batch, tokens)
        mask = tf.not_equal(inputs, 0)
        # (batch, tokens, edim)
        x = self.embedding_layer(inputs)
        # (batch, edim)
        x = self.masked_pooling(x, mask)
        if hasattr(self, "mlp"):
            x = self.mlp(x, training=training)
            
        return x


class AttnDSSM(nn.Model):

    def __init__(
        self, 
        vocab_size:int,
        embedding_dim: int,
        sentence_emb_dim: int,
        dense_layer_nodes:List[int] = None, 
        activation:str = "relu", 
        batch_norm:bool = True,
        dropout_rate:float = None
    ):
        super().__init__()
        self.embedding_layer = nn.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.sentence_embedder = nn.layers.GRU(sentence_emb_dim, return_sequences=True)
        self.attention_layer = AttentionLayer()
        if dense_layer_nodes:
            self.mlp = create_tower(hidden_layer_dims=dense_layer_nodes[:-1],
                                    output_units=dense_layer_nodes[-1],
                                    activation=activation,
                                    output_activation=None,
                                    input_batch_norm=False,
                                    use_batch_norm=batch_norm,
                                    dropout=dropout_rate)

    def call(self, inputs, training):
        mask = tf.not_equal(inputs, 0)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        x = self.embedding_layer(inputs)
        x = self.sentence_embedder(x)
        attn = self.attention_layer(x, mask)
        x = tf.reduce_sum(attn * x, axis=1)  
        if hasattr(self, "mlp"):
            x = self.mlp(x, training=training)
        return x


class USE(nn.Model):

    def __init__(
        self, 
        vocab_size:int,
        embedding_dim: int,
        dff:int=2048,
        num_attn_heads:int= 8,
        num_attn_layers:int = 4,
        dropout_rate:float=0.1
    ):
        super().__init__()
        self.d_model = embedding_dim

        self.embedding_layer = nn.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(config.MAX_TOKENS, self.d_model)
        self.enc_layers = [
            EncoderLayer(d_model=self.d_model, num_heads=num_attn_heads, dff=dff, rate=dropout_rate)
            for _ in range(num_attn_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        
        mask = tf.not_equal(inputs, 0)
        seq_lengths = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1, keepdims=True)
        seq_len_not_masked = tf.shape(inputs)[1]

        x = self.embedding_layer(inputs)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len_not_masked, :]

        x = self.dropout(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training, mask[:, tf.newaxis, tf.newaxis, :])
        # (batch_size, input_seq_len, d_model) -> # (batch_size, d_model)
        # additional masking is needed, as padded tokens receive nonzero embeddings
        x = tf.reduce_sum(tf.ragged.boolean_mask(x, mask),axis=1)
        x /= tf.sqrt(seq_lengths)

        return x  


class KNRM(nn.Model):

    def __init__(
            self, 
            vocab_size:int,
            embedding_dim: int,
            kernel_num: int, 
            sigma = 0.1,
            exact_sigma = 0.001,
            nodes: list = [1],
            inner_activations: str = None,
            output_activation: str = "sigmoid",
            dropout_rate:float = None,
            batch_norm:bool = False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = nn.layers.Embedding(vocab_size, embedding_dim)
        self.mlp = create_tower(hidden_layer_dims=nodes[:-1],
                                output_units=nodes[-1],
                                activation=inner_activations,
                                output_activation=output_activation,
                                use_batch_norm=batch_norm,
                                dropout=dropout_rate)

        self.kernel_pooling = KNRMLayer(self.embedding_layer,
                                    kernel_num,
                                    sigma,
                                    exact_sigma,
                                    sequence_dim=None)

    def call(self, inputs, training):
        """
        B = batch size
        Q = query length
        D = document length
        BS = bin size
        E = embedding size
        K = kernel_num
        """
        # [B, Q]
        query = inputs[config.QUERY_COL]
        # shape = [B, Q]
        mask_q = tf.cast(tf.not_equal(query, 0), tf.float32)

        # [B, Q, K]
        kde = self.kernel_pooling(inputs, training)
        # [B, Q, K] -> [B, Q, 1]
        scores = self.mlp(kde, training=training)
        # [B, 1]
        aggregated_scores = tf.reduce_sum(mask_q[...,tf.newaxis] * scores, axis=1)       
        return aggregated_scores
        

class KNRMwAttn(nn.Model):

    def __init__(
            self, 
            vocab_size:int,
            embedding_dim: int,
            kernel_num: int, 
            sigma = 0.1,
            exact_sigma = 0.001,
            nodes: list = [1],
            inner_activations: str = None,
            output_activation: str = "sigmoid",
            dropout_rate:float = None,
            batch_norm:bool = False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = nn.layers.Embedding(vocab_size, embedding_dim)
        self.mlp = create_tower(hidden_layer_dims=nodes[:-1],
                                output_units=nodes[-1],
                                activation=inner_activations,
                                output_activation=output_activation,
                                use_batch_norm=batch_norm,
                                dropout=dropout_rate)

        self.kernel_pooling = KNRMLayer(self.embedding_layer,
                                    kernel_num,
                                    sigma,
                                    exact_sigma,
                                    sequence_dim=None)

        self.att_layer = AttentionLayer()

    def call(self, inputs, training):
        """
        B = batch size
        Q = query length
        D = document length
        BS = bin size
        E = embedding size
        K = kernel_num
        """
        # [B, Q]
        query = inputs[config.QUERY_COL]

        # shape = [B, Q]
        mask_q = tf.cast(tf.not_equal(query, 0), tf.float32)

        # shape = [B, Q, 1]
        atten_mask = tf.expand_dims(mask_q, axis=2)

        # [B, Q, E]
        q_embed = self.embedding_layer(query)

        # [B,Q,K]
        kde = self.kernel_pooling(inputs, training)

        # [B, Q, K] -> [B, Q, 1]
        scores = self.mlp(kde, training=training)
 
        # shape = [B, Q, 1]
        attention_probs = self.att_layer(q_embed, atten_mask)
        # [B, 1]
        aggregated_scores = tf.reduce_sum(attention_probs * scores, axis=1)  

        return aggregated_scores