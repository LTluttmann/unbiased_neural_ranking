import tensorflow as tf
import tensorflow.keras as nn

from mt.config import config
from mt.data.dataset.utils import merge_first_two_dims
from mt.models.losses import bpr_max, softmax_crossentropy_loss, softmax_crossentropy_loss_new

import abc


class KerasBaseTrainer(abc.ABC, nn.Model):
    def __init__(self, encoder, graph_exec:bool = True, eval_metrics = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.graph_exec = graph_exec
        self.eval_metrics = eval_metrics

    @classmethod
    def load_base_model(cls, path, *args, **kwargs):
        encoder = nn.models.load_model(path, compile=False)
        return cls(encoder, *args, **kwargs)

    @abc.abstractmethod
    def _train_on_batch(self, inputs):
        pass

    @abc.abstractmethod
    def _validate_on_batch(self, inputs):
        pass

    @tf.function
    def _tf_train_on_batch(self, inputs):
        return self._train_on_batch(inputs)

    @tf.function
    def _tf_validate_on_batch(self, inputs):
        return self._validate_on_batch(inputs) 

    def train_step(self, inputs):
        if self.graph_exec:
            return self._tf_train_on_batch(inputs)
        else:
            return self._train_on_batch(inputs)

    def test_step(self, inputs):
        if self.graph_exec:
            return self._tf_validate_on_batch(inputs)
        else:
            return self._validate_on_batch(inputs)   

    def update_metrics(self, label, pred) -> dict:

        if not self.eval_metrics:
            return dict()

        for metric in self.eval_metrics:

            # metric_mask = tf.cast(tf.not_equal(label, -1), tf.float32)

            metric.update_state(label, pred, sample_weight=None)

        return {m.name: m.result() for m in self.eval_metrics}


class EfficientContrastiveLearner(KerasBaseTrainer):

    def __init__(
        self,         
        encoder: nn.Model, 
        loss = softmax_crossentropy_loss,
        normalize = True,
        graph_exec:bool = True, 
        subsample_size:int = -1,
        **kwargs
    ):
        super().__init__(encoder, graph_exec)
        self.loss_fn = loss(**kwargs)
        self.dot = nn.layers.Dot(axes=[-1,-1], normalize=normalize)
        self.subsample_size = subsample_size

        # input specs
        self.cos_sim.get_concrete_function(
            a=tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            b=tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        )
        self.cos_sim.get_concrete_function(
            a=tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            b=tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
        )

    @tf.function
    def cos_sim(self, a, b, axis=-1):
        a = tf.linalg.l2_normalize(a, axis=axis)
        b = tf.linalg.l2_normalize(b, axis=axis)
        return tf.reduce_sum(a * b, axis=axis)

    @tf.function
    def subsample(self, logits):
        # [BS, (P+N)]
        batch_size = tf.shape(logits)[0]
        num_docs = tf.shape(logits)[1]
        # extract positive logits which is the first column -> [BS, 1]
        pos_logits = tf.expand_dims(tf.gather(logits, 0, axis=1), 1)
        # assign very large negative number to clicked documents
        mask_values = tf.ones((batch_size,1), dtype=tf.float32) * -1e9
        # remove first column from tensor (clicked docs) and replace with large negatives
        masked_logits = tf.concat((mask_values, logits[:,1:]),axis=-1)
        # apply gumbal max trick for sampling negatives
        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits),0,1)))
        # since we take the max after adding gumbal random var, we will never draw a clicked doc
        _, indices = tf.nn.top_k(masked_logits + z, tf.minimum(num_docs, self.subsample_size) )
        # get the original logits from the sampled indices (NOTE: cant use values of top_k func, as it returns logits+z)
        neg_logits = tf.gather(logits, indices, batch_dims=1, axis=1)
        # concat positives and negatives -> [BS, (P+N)]
        logits = tf.concat((pos_logits, neg_logits), axis=-1)
        return logits

    def call(self, inputs, training=True):
        """
        BS = Batch size
        P = Number of positive examples per query (always fixed to 1)
        N = Number of negative examples per query
        E = Embedding / feature size
        T = tokens
        """
        # [BS, T]
        query = inputs[config.QUERY_COL]
        # [BS, P+N, T]
        docs = inputs[config.PRODUCT_TITLE_COL]
        batch_size = tf.shape(docs)[0]
        num_docs = tf.shape(docs)[1]
        # [BS * (P+N), T], reduce to 2 dimension to be processable by the encoder
        if isinstance(docs, tf.RaggedTensor):
            docs = docs.merge_dims(0, 1)
        else:
            docs, _ = merge_first_two_dims(docs)
        # [BS, E]
        query_emb = self.encoder(query, training=training)
        # [BS*(P+N), E]
        doc_embs = self.encoder(docs, training=training)
        # [BS, (P+N), E]
        doc_embs = tf.reshape(doc_embs, (batch_size, num_docs, -1))
        # Query embeddings: [BS, E] -> [BS, 1, E]
        # logits output shape: [BS, (P+N)]
        logits = self.dot([query_emb, doc_embs])
        logits = tf.cond(tf.math.logical_and(training, tf.not_equal(self.subsample_size, -1)), 
                         lambda: self.subsample(logits), 
                         lambda: logits)
        return logits

    def _train_on_batch(self, inputs):
        y_true = inputs.get(config.CLICK_COL)
        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            loss = self.loss_fn(y_true, logits)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": tf.reduce_mean(loss)}

    def _validate_on_batch(self, inputs):
        y_true = inputs.get(config.CLICK_COL)
        logits = self(inputs, training=False)
        loss = self.loss_fn(y_true, logits)
        return {"loss": tf.reduce_mean(loss)}


class InBatchLearner(KerasBaseTrainer):

    def __init__(
        self, 
        encoder: nn.Model, 
        num_negatives:int=1, 
        loss = nn.losses.CategoricalCrossentropy(from_logits=True),
        temperature: float = 1.0,
        graph_exec: bool = True, 
        alpha_bounds: list = [0.,0.],
        metrics: list = None,
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(encoder, graph_exec, eval_metrics=metrics)
        self.num_negatives = num_negatives
        self.alpha_bounds = alpha_bounds
        self.strategy = self.get_strategy()
        self.loss_fn = loss
        self.temperature = temperature
        self.val_loss_fn = nn.losses.CategoricalCrossentropy(from_logits=True)
        self.normalize = normalize

    def get_strategy(self):
        if self.num_negatives == -1:
            return self.batch_all
        else:
            if all([x==0 for x in self.alpha_bounds]):
                return self.batch_hard
            else:
                return self.batch_self_reinforcing_hard


    @tf.function
    def cos_sim(self, a, b, axis=-1):
        a = tf.linalg.l2_normalize(a, axis=axis)
        b = tf.linalg.l2_normalize(b, axis=axis)
        return tf.reduce_sum(a * b, axis=axis)

    def call(self, inputs, training=True):
        """
        Q = num query tokens
        D = num product tokens
        E = embedding dimension
        """
        # [BS, Q]
        query = inputs.get(config.QUERY_COL)
        # [BS, D]
        docs = inputs.get(config.PRODUCT_TITLE_COL)
        # [BS, E]
        query_emb = self.encoder(query, training=training)
        doc_embs = self.encoder(docs, training=training)
        if training:
            labels, logits = self.strategy(query_emb, doc_embs)
        else:
            labels, logits = self.batch_all(query_emb, doc_embs)
        return labels, logits

    @tf.function
    def _get_k_hardest_negatives(self, qe, de, k=1, return_indices=False):
        """get the k negatives which are closest to the query"""
        batch_size = tf.shape(qe)[0]
        # [BS, E]
        if self.normalize:
            qe = tf.linalg.l2_normalize(qe, axis=-1)
            de = tf.linalg.l2_normalize(de, axis=-1)
        # [1, BS, E]
        qe_ = tf.expand_dims(qe, 0)
        # [BS, 1, E]
        de_ = tf.expand_dims(de, 1)
        # [BS, BS]
        similarity = tf.reduce_sum(qe_ * de_, axis=-1)
        # we need to mask the similarities on the diagonal, since those are the
        # the similarity scores of the positive examples.
        # the lower bound of cosine similarity is -1, hence use it for masking (we take max next)
        masked_similarities = tf.linalg.set_diag(similarity, -tf.ones(batch_size) * 1e9)
        # [BS, k]
        values, indices = tf.math.top_k(masked_similarities, k=k, sorted=False)
        if return_indices:
            return tf.gather(de, indices, batch_dims=1, axis=0)
        else:
            return values

    @tf.function
    def batch_all(self, qe, de):
        batch_size = tf.shape(qe)[0]
        # [BS, E]
        if self.normalize:
            qe = tf.linalg.l2_normalize(qe, axis=-1)
            de = tf.linalg.l2_normalize(de, axis=-1)
        # [1, BS, E]
        qe_ = tf.expand_dims(qe, 0)
        # [BS, 1, E]
        de_ = tf.expand_dims(de, 1)
        # [BS, BS]
        logits = tf.reduce_sum(qe_ * de_, axis=-1)
        labels = tf.one_hot(tf.range(batch_size), depth=batch_size)

        return labels, logits


    @tf.function
    def batch_hard(self, qe, de):
        batch_size = tf.shape(qe)[0]
        # [BS, E] -> [BS]
        positive_similarities = self.cos_sim(qe, de, axis=-1)
        # [BS, E] -> [BS, k]
        negative_similarities = self._get_k_hardest_negatives(qe, de, k=self.num_negatives)
        # [BS] -> [BS, 1]
        positive_similarities = tf.reshape(positive_similarities, (-1, 1))
        # [BS, k+1]
        logits = tf.concat((positive_similarities, negative_similarities), axis=-1)
        # [BS] -> contains the index of logits from true class for sparse categorical ce
        labels = tf.one_hot(tf.zeros(batch_size, dtype=tf.int64), depth=self.num_negatives+1)
        return labels, logits

    @tf.function
    def batch_self_reinforcing_hard(self, qe, de):
        batch_size = tf.shape(qe)[0]
        # [BS, E] -> [BS]
        positive_similarities = self.cos_sim(qe, de, axis=-1)
        # [BS, E] -> [BS, k, E]
        hard_negatives = self._get_k_hardest_negatives(qe, de, k=self.num_negatives, return_indices=True)
        # [BS, k]
        alpha = tf.random.uniform(hard_negatives.get_shape().as_list()[:-1], *self.alpha_bounds)
        # [BS, k, 1]
        alpha = tf.expand_dims(alpha, axis=-1)
        # [BS, k, E]
        hard_negatives = alpha * tf.expand_dims(de, 1) + (1-alpha) * hard_negatives

        negative_similarities = tf.keras.layers.Dot(axes=[2,2], normalize=True)([tf.expand_dims(qe, 1), hard_negatives])
        negative_similarities = tf.squeeze(negative_similarities, 1)
        # [BS] -> [BS, 1]
        positive_similarities = tf.reshape(positive_similarities, (-1, 1))
        # [BS, k+1]
        logits = tf.concat((positive_similarities, negative_similarities), axis=-1)
        # [BS] -> contains the index of logits from true class for sparse categorical ce
        labels = tf.one_hot(tf.zeros(batch_size), depth=self.num_negatives+1)
        return labels, logits

    def _train_on_batch(self, inputs):
        with tf.GradientTape() as tape:
            labels, logits = self(inputs, training=True)
            scaled_logits = logits / self.temperature
            loss = self.loss_fn(labels, scaled_logits)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": tf.reduce_mean(loss)}

    def _validate_on_batch(self, inputs):
        labels, logits = self(inputs, training=False)
        loss = self.val_loss_fn(labels, logits)
        metrics = self.update_metrics(labels, logits)
        return {"loss": tf.reduce_mean(loss), **metrics}