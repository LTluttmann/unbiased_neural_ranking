import tensorflow as tf
import tensorflow.keras as nn

from mt.config import config
from mt.data.dataset.utils import merge_first_two_dims
from mt.models.mmoe import MMoE
from mt.models.batch_norm import BatchNormalization

from typing import List, Optional, Dict, Any


class MultiTaskScoringFn(nn.layers.Layer):
    def __init__(self,
                 num_tasks:int, 
                 num_experts:int=4,
                 expert_units:int=512,
                 tower_units:List[int]=[512],
                 output_units:int=1,
                 dropout:float=0.5,
                 trainable:bool=True, 
                 name=None, 
                 dtype=None, 
                 dynamic=False, 
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.mmoe_layers = MMoE(units=expert_units, num_experts=num_experts, num_tasks=num_tasks)

        self.output_layers = []
        self.towers = []
        # Build tower layer from MMoE layer
        for _ in range(num_tasks):
            tower = create_tower(tower_units, output_units, activation="relu", use_batch_norm=True, dropout=dropout)
            self.towers.append(tower)

    def call(self, x, training=True):

        x, dim = merge_first_two_dims(x)

        scores = []
        mmoe = self.mmoe_layers(x, training=training)
        for i, x in enumerate(mmoe):
            x = self.towers[i](x, training=training)
            x = tf.reshape(x, (-1, dim))
            scores.append(x)
        return scores


class PbkClassification(nn.layers.Layer):
    def __init__(self, classifier: nn.Model, pbks: List[str], trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        # NOTE: set trainable to false to freeze all sublayers
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.classifier = classifier
        self.pbks = pbks
        self.pbk_lookup = nn.layers.StringLookup(vocabulary=self.pbks)

    def call(self, inputs: dict):
        # [BS, T]
        q = inputs[config.QUERY_COL]
        # [BS, C]
        class_pred = self.classifier(q, training=False)
        # [BS, 1, C]
        class_pred = class_pred[:, tf.newaxis, :]
        # [BS, S]
        pbk_looked_up = self.pbk_lookup(inputs[config.PBK_COL])
        # [BS, S, C]
        pbk_one_hot = tf.one_hot(pbk_looked_up, depth=len(self.pbks)+1)
        # [BS, S, C] * [BS, 1, C] -> [BS, S, C] -> sum over classes: [BS, S, 1] (keepdims for concat)
        class_feature = tf.reduce_sum(pbk_one_hot * class_pred, axis=-1, keepdims=False)
        # mask values of padded items
        mask = tf.not_equal(tf.reduce_sum(inputs[config.PRODUCT_TITLE_COL], axis=-1), 0)
        # class_feature = tf.where(mask[...,tf.newaxis], class_feature, -1.0)
        class_feature = tf.where(mask, class_feature, -1.0)
        return tf.expand_dims(class_feature, axis=-1)


class SemanticMatchingScorer(nn.layers.Layer):
    def __init__(self, encoder: nn.Model, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        # NOTE: set trainable to false to freeze all sublayers
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.encoder = encoder

    def call(self, inputs: dict):
        bs = tf.shape(inputs[config.PRODUCT_TITLE_COL])[0]
        seq_len = tf.shape(inputs[config.PRODUCT_TITLE_COL])[1]
        token_len = tf.shape(inputs[config.PRODUCT_TITLE_COL])[2]
        # [BS, T] -> [BS, E]
        q_emb = self.encoder(inputs[config.QUERY_COL], training=False)
        emb_dim = tf.shape(q_emb)[1]
        # [BS, S, T] -> [BS*S, T] -> [BS*S, E]
        doc_emb = self.encoder(tf.reshape(inputs[config.PRODUCT_TITLE_COL], (-1, token_len)), training=False)
        # [BS*S, E] -> [BS, S, E]
        doc_emb = tf.reshape(doc_emb, (bs, seq_len, emb_dim))
        # [BS, E] * [BS, S, E] -> [BS, S]
        sms = nn.layers.Dot(axes=[1,2], normalize=True)([q_emb, doc_emb])
        # mask values of padded items
        sms = tf.where(tf.not_equal(tf.reduce_sum(inputs[config.PRODUCT_TITLE_COL], axis=-1), 0), sms, -1.0)
        # [BS, S] -> [BS, S, 1] (expand for concat)
        return tf.expand_dims(sms, axis=-1)


class KNRMLayer(nn.layers.Layer):

    def __init__(
            self, 
            embedding_layer,
            kernel_num: int = 11, 
            sigma = 0.1,
            exact_sigma = 0.001,
            sequence_dim = None,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.sequence_dim = sequence_dim

        mu = 1. / (kernel_num - 1) + (2. * tf.range(kernel_num, dtype=tf.float32)) / (kernel_num - 1) - 1.0
        self.sigma = tf.where(tf.greater(mu, 1.0), exact_sigma, sigma)
        self.mu = tf.where(tf.greater(mu, 1.0), 1.0, mu)
        
    def _kernel_pooling(self, q_embed, d_embed, mask_q, mask_d):
        """
        B = batch size
        Q = query length
        D = document length
        BS = bin size
        E = embedding size
        K = kernel_num
        """
        # [B, Q, D]
        mask_q = tf.expand_dims(mask_q, 2)
        mask_d = tf.expand_dims(mask_d, 1)
        mask_qd = tf.multiply(mask_q, mask_d)
        # [B, Q, D, 1]
        mask_qd = tf.expand_dims(mask_qd, -1)

        # [B, Q, E] * [B, D, E] -> [B, Q, D] // NOTE: tested and works fine
        sim = tf.keras.layers.Dot(axes=[-1, -1], normalize=True)([q_embed, d_embed])

        # compute gaussian
        # [B, Q, D, 1] 
        rs_sim = tf.expand_dims(sim, -1)
        # compute Gaussian scores of each kernel
        # [B, Q, D, K] 
        tmp = tf.exp(-tf.square(rs_sim - self.mu) / (tf.square(self.sigma) * 2))
        # mask those non-existing words.
        tmp = tmp * mask_qd
        # [B, Q, K]
        kde = tf.reduce_sum(tmp, [-2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01 
        # [B, Q, K]
        return kde      

    def _kernel_pooling_on_batch_of_sequences(self, q_embed, d_embed, mask_q, mask_d):
        """
        B = batch size
        Q = query length
        D = document length
        PN = number of examples per instance (positives + negatives)
        BS = bin size
        E = embedding size
        K = kernel_num
        """
        # [B, PN, Q, D]
        mask_qd = tf.multiply(mask_q[:, tf.newaxis, :, tf.newaxis],
                              mask_d[:, :, tf.newaxis, :])
        # [B, PN, Q, D, 1]
        mask_qd = tf.expand_dims(mask_qd, -1)   

        # [B, Q, PN, D] -> [B, PN, Q, D] // NOTE: tested and works fine
        sim = tf.keras.layers.Dot(axes=[-1, -1], normalize=True)([q_embed, d_embed])
        sim = tf.transpose(sim, [0,2,1,3])
        # compute gaussian
        # [B, PN, Q, D, 1] 
        rs_sim = tf.expand_dims(sim, -1)
        # compute Gaussian scores of each kernel
        # [B, PN, Q, D, K] 
        tmp = tf.exp(-tf.square(rs_sim - self.mu) / (tf.square(self.sigma) * 2))
        # mask those non-existing words.
        tmp = tmp * mask_qd
        # [B, PN, Q, D, K] -> [B, PN, Q, K]
        kde = tf.reduce_sum(tmp, [-2])
        # kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01 
        kde = tf.math.log1p(kde)
        # [B, PN, Q, K]
        return kde

    def call(self, inputs, training):
        # [B, Q]
        query = inputs[config.QUERY_COL]
        # [B, D]
        documents = inputs[config.PRODUCT_TITLE_COL]

        # shape = [B, Q]
        mask_q = tf.cast(tf.not_equal(query, 0), tf.float32)
        # shape = [B, L]
        mask_d = tf.cast(tf.not_equal(documents, 0), tf.float32)
        
        # [B, Q, E]
        q_embed = self.embedding_layer(query, training=training)
        # [B, D, E]
        d_embed = self.embedding_layer(documents, training=training)
        if self.sequence_dim is not None:
            return self._kernel_pooling_on_batch_of_sequences(q_embed, d_embed, mask_q, mask_d)
        else:
            return self._kernel_pooling(q_embed, d_embed, mask_q, mask_d)


class MatchHistogram(nn.layers.Layer):
    def __init__(self, encoder, bin_size, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.embedding_model = encoder 
        self.bin_size = bin_size

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="query_embedding"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="document_embedding"),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="document_mask")])
    def _match_histogram_batched(self, query_emb, doc_emb, mask):
        """
        BS = Batch size
        H = histogram bin size
        A = anchor sequence length
        D = document sequence length
        E = embedding dimension
        """
        batch_size = tf.shape(doc_emb)[0]
        anchor_seq_size = tf.shape(query_emb)[1]
        doc_seq_size = tf.shape(doc_emb)[1]
        # reduce precision to reduce risks of similar tokens not matching to 1
        # [BS, A, E]
        query_emb = tf.cast(query_emb, tf.float16)
        # [BS, D, E]
        doc_emb = tf.cast(doc_emb, tf.float16)
        # normalize
        query_emb = tf.linalg.l2_normalize(query_emb, axis=-1)
        doc_emb = tf.linalg.l2_normalize(doc_emb, axis=-1)
        # Docs: [BS, D, E] -> [BS, 1, D, E]; Query: [BS, A, E] -> [BS, A, 1, E]
        # [BS, 1, D, E] * [BS, A, 1, E] -> [BS, A, D, E]
        # Sum: [BS, A, D, E] -> [BS, A, D]
        da_cos_similarities = tf.reduce_sum(tf.expand_dims(doc_emb, 1) * tf.expand_dims(query_emb, 2), axis=-1)
        # [BS, A, D]
        da_bin_hashes = tf.cast((da_cos_similarities + 1.) / 2. * (self.bin_size-1.), tf.int32)
        # [BS*A, D]
        da_bin_hashes = tf.reshape(da_bin_hashes, (-1, doc_seq_size))
        # [BS*A, H]
        da_bin_counts = tf.math.bincount(da_bin_hashes, weights=mask, minlength=self.bin_size, maxlength=self.bin_size, dtype=tf.float32, axis=-1)
        # [BS, A, H]
        da_bin_counts = tf.reshape(da_bin_counts, (batch_size, anchor_seq_size, self.bin_size))
        # [BS, A, H]
        log_bin_counts = tf.math.log(da_bin_counts+1)
        return log_bin_counts

    # @tf.function
    def call(self, inputs: dict):
        # copy needed as python input is changed (tf error otherwise)
        x = inputs.copy()
        
        # [BS, A]
        query = x[config.QUERY_COL]
        if isinstance(query, tf.RaggedTensor):
            query = query.to_tensor()

        batch_size = tf.shape(query)[0]
        anchor_seq_size = tf.shape(query)[1]

        # [BS, P+N, D]
        docs = x[config.PRODUCT_TITLE_COL]
        if isinstance(docs, tf.RaggedTensor):
            docs = docs.to_tensor()

        num_examples = tf.shape(docs)[1]
        # [BS*(P+N), D]
        docs = tf.reshape(docs, (batch_size*num_examples, -1))
        # query = tf.repeat(query, [num_examples], axis=0)      

        # mask padded document entries. NOTE: query does not have to be masked
        # here, since this happens in the attention layer of the drmm model
        mask = tf.not_equal(docs, 0)
        mask = tf.cast(tf.repeat(mask, [anchor_seq_size], axis=0), tf.float32)

        query_emb = self.embedding_model(query)
        query_emb = tf.repeat(query_emb, [num_examples], axis=0)    
        doc_emb = self.embedding_model(docs)

        log_bin_counts = self._match_histogram_batched(query_emb, doc_emb, mask)

        log_bin_counts = tf.reshape(log_bin_counts, (batch_size, num_examples, anchor_seq_size, self.bin_size))
        return log_bin_counts


class EncoderLayer(nn.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        head_size_d = d_model // num_heads
        
        self.mha = nn.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size_d)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output = self.mha(x, x, x, mask, return_attention_scores=False)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class AttentionLayer(nn.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.layers.Dense(1, activation=activation, use_bias=False)
        
    def call(self, inputs, mask):
        # [B, LS, E] -> [B, LS, 1]
        dense_input = self.dense(inputs)
        dense_input = dense_input + (1.0 - mask) * -10_000.0
        # [B, LS, 1]
        
        # softmax
        e = tf.exp(dense_input)
        s = tf.reduce_sum(e, axis=1, keepdims=True) # + 1e-16
        output = e / s
        # output = tf.nn.softmax(dense_input, axis=1)
        # [B, LS, 1]
        return output


class MultiLayerPerceptron(nn.layers.Layer):

    def __init__(self,
                 hidden_layer_dims: List[int],
                 output_units: int = None,
                 activation: str = None,
                 output_activation: str = None,
                 input_batch_norm: bool = False,
                 use_batch_norm: bool = True,
                 dropout: float = 0.5,
                 **kwargs: Dict[Any, Any]):
        """general purpose mlp"""
        super().__init__(**kwargs)
        # dense layers
        self.layers = [nn.layers.Dense(i) for i in hidden_layer_dims]
        self.activation = nn.layers.Activation(activation)
        # output 
        if output_units:
            self.output_layer = nn.layers.Dense(output_units)
            self.output_activation = nn.layers.Activation(output_activation)
        # dropout
        self.dropout = [nn.layers.Dropout(dropout) if dropout else None for _ in hidden_layer_dims]
        # batch norm
        # NOTE these layers do not work with masks. When sampling is applied, there is no need for 
        # masking. However, in validation we might want to consider the whole Serp and the behavior 
        # there will be unconsistent compared to training. 
        self.batch_norm = [nn.layers.BatchNormalization() if use_batch_norm else None for _ in hidden_layer_dims]
        if input_batch_norm:
            self.input_norm = nn.layers.BatchNormalization()

    def call(self, x, training=False):
        # batch normalize input
        if hasattr(self, "input_norm"):
            x = self.input_norm(x, training=training)

        # call hidden layers
        for dense, bn, dropout in zip(self.layers, self.batch_norm, self.dropout):
            # h = (WX+b)
            x = dense(x)
            # batch norm before activation
            if bn is not None:
                x = bn(x, training=training)
            # sigma(h)
            x = self.activation(x)
            # lastly, dropout
            if dropout is not None:
                x = dropout(x, training=training)
        # call output layer
        if hasattr(self, "output_layer"):
            x = self.output_layer(x)
            x = self.output_activation(x)
        
        return x


def create_tower(hidden_layer_dims: List[int],
                 output_units: int = None,
                 activation: str = None,
                 output_activation: str = None,
                 input_batch_norm: bool = False,
                 use_batch_norm: bool = True,
                 batch_norm_moment: float = 0.999,
                 dropout: float = 0.5,
                 name: Optional[str] = None,
                 mask_value = None,
                 **kwargs: Dict[Any, Any]):
    """Creates a feed-forward network as `tf.keras.Sequential`.
    It creates a feed-forward network with batch normalization and dropout, and
    optionally applies batch normalization on inputs.
    Returns:
    A `tf.keras.Sequential` object.
    """
    model = tf.keras.Sequential(name=name)
    # Input batch normalization.
    if input_batch_norm:
        if mask_value is not None:
            model.add(BatchNormalization(mask_value=mask_value))
        else:
            model.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
    for layer_width in hidden_layer_dims:
        model.add(tf.keras.layers.Dense(units=layer_width), **kwargs)
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization(momentum=batch_norm_moment))
        model.add(tf.keras.layers.Activation(activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    if output_units:
        model.add(tf.keras.layers.Dense(units=output_units), **kwargs)
        model.add(tf.keras.layers.Activation(activation=output_activation))
    return model


class ToDense(nn.layers.Layer):  # pylint: disable=g-classes-have-attributes
    """Layer that makes padding and masking a Composite Tensors effortless.
    The layer takes a RaggedTensor or a SparseTensor and converts it to a uniform
    tensor by right-padding it or filling in missing values.
    Example:
    ```python
    x = tf.keras.layers.Input(shape=(None, None), ragged=True)
    y = tf_text.keras.layers.ToDense(mask=True)(x)
    model = tf.keras.Model(x, y)
    rt = tf.RaggedTensor.from_nested_row_splits(
      flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
      nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
    model.predict(rt)
    [[[10, 11, 12,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
     [[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
     [[ 0,  0,  0,  0], [13, 14,  0,  0], [15, 16, 17, 18], [19,  0,  0,  0]]]
    ```
    Args:
      pad_value: A value used to pad and fill in the missing values. Should be a
        meaningless value for the input data. Default is '0'.
      mask: A Boolean value representing whether to mask the padded values. If
        true, no any downstream Masking layer or Embedding layer with
        mask_zero=True should be added. Default is 'False'.
      shape: If not `None`, the resulting dense tensor will be guaranteed to have
        this shape. For RaggedTensor inputs, this is passed to `tf.RaggedTensor`'s
        `to_tensor` method. For other tensor types, a `tf.ensure_shape` call is
        added to assert that the output has this shape.
      **kwargs: kwargs of parent class.
    Input shape: Any Ragged or Sparse Tensor is accepted, but it requires the type
      of input to be specified via the Input or InputLayer from the Keras API.
    Output shape: The output is a uniform tensor having the same shape, in case of
      a ragged input or the same dense shape, in case of a sparse input.
    """

    def __init__(self, pad_value=0, mask=False, shape=None, **kwargs):
        super(ToDense, self).__init__(**kwargs)

        self._pad_value = pad_value
        self._mask = mask
        self._shape = shape
        self._compute_output_and_mask_jointly = True
        self._supports_ragged_inputs = True
        self.trainable = False
        self.masking_layer = nn.layers.Masking(
            mask_value=self._pad_value)

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            # Convert the ragged tensor to a padded uniform tensor
            outputs = inputs.to_tensor(
                default_value=self._pad_value, shape=self._shape)
        elif isinstance(inputs, tf.sparse.SparseTensor):
            # Fill in the missing value in the sparse_tensor
            outputs = tf.sparse.to_dense(inputs, default_value=self._pad_value)
            if self._shape is not None:
                outputs = tf.ensure_shape(outputs, shape=self._shape)
        elif isinstance(inputs, tf.Tensor):
            outputs = inputs
            if self._shape is not None:
                outputs = tf.ensure_shape(outputs, shape=self._shape)
        else:
            raise TypeError('Unexpected tensor type %s' %
                            type(inputs).__name__)

        if self._mask:
            outputs = self.masking_layer(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):

        config = {
            'pad_value': self._pad_value,
            'mask': self._mask,
            'shape': self._shape,
        }
        base_config = super(ToDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ContrastiveLossLayer(nn.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negatives):
        tf.assert_equal(tf.rank(anchor), 2)
        tf.assert_equal(tf.rank(positive), 2)
        tf.assert_equal(tf.rank(negatives), 3)

        # adjust dimensionality of tensors so that they fit
        positive = tf.expand_dims(positive, 0)
        anchor = tf.expand_dims(anchor, 0)
        # formula taken from van gysel
        pos_probs = tf.nn.sigmoid(tf.reduce_sum(
            tf.multiply(anchor, positive), axis=-1))
        neg_probs = tf.nn.sigmoid(tf.reduce_sum(
            tf.multiply(anchor, negatives), axis=-1))
        loss = tf.squeeze(tf.math.log(pos_probs)) + \
            tf.reduce_sum(tf.math.log(1-neg_probs + 1e-6), axis=0)
        return tf.negative(loss)
