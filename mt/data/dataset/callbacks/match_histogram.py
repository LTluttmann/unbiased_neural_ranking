from mt.data.dataset.dataset_ops import BaseDatasetOp
from mt.config import config

import tensorflow as tf


# NOTE DEPRECATED! Implemented as layer in mt.models.layers
class HistogramCallback(BaseDatasetOp):
    def __init__(self, embedding_model: tf.keras.Model, pairwise: bool = True, bin_size:int = 30, on_batch:bool = True) -> None:
        super().__init__(on_batch=on_batch)
        self.embedding_model = embedding_model
        self.pairwise = pairwise
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

    @tf.function
    def _batch_operation_pairwise(self, inputs: dict):
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
        query = tf.repeat(query, [num_examples], axis=0)      

        # mask padded document entries. NOTE: query does not have to be masked
        # here, since this happens in the attention layer of the drmm model
        mask = tf.not_equal(docs, 0)
        mask = tf.cast(tf.repeat(mask, [anchor_seq_size], axis=0), tf.float32)

        query_emb = self.embedding_model(query)
        doc_emb = self.embedding_model(docs)

        log_bin_counts = self._match_histogram_batched(query_emb, doc_emb, mask)

        log_bin_counts = tf.reshape(log_bin_counts, (batch_size, num_examples, anchor_seq_size, self.bin_size))
        x.update({config.MATCH_HIST_COL: log_bin_counts})
        return x


    # @tf.function(input_signature=[{
    #     config.QUERY_COL: tf.TensorSpec(shape=[None, None], dtype=tf.int64),
    #     config.PRODUCT_TITLE_COL: tf.TensorSpec(shape=[None, None], dtype=tf.int64)
    # }])
    @tf.function
    def _batch_operation_pointwise(self, inputs: dict):
        # copy needed as python input is changed (tf error otherwise)
        x = inputs.copy()
        # [BS, A]
        query = x[config.QUERY_COL]
        # [BS, D]
        doc = x[config.PRODUCT_TITLE_COL]
        
        anchor_seq_size = tf.shape(query)[1]

        # mask padded document entries. NOTE: query does not have to be masked
        # here, since this happens in the attention layer of the drmm model
        mask = tf.not_equal(doc, 0)
        mask = tf.cast(tf.repeat(mask, [anchor_seq_size], axis=0), tf.float32)
        # [BS, A, E]
        query_emb = self.embedding_model(query)
        # [BS, D, E]
        doc_emb = self.embedding_model(doc)

        log_bin_counts = self._match_histogram_batched(query_emb, doc_emb, mask)
                
        x.update({config.MATCH_HIST_COL: log_bin_counts})
        return x

    def call_on_batch(self, inputs: dict):
        if self.pairwise:
            return self._batch_operation_pairwise(inputs)
        else:
            return self._batch_operation_pointwise(inputs)

    def call_on_single_example(self, inputs):
        x = inputs.copy()
        
        query = x[config.QUERY_COL]
        docs = x[config.PRODUCT_TITLE_COL]
        
        query_emb = self.embedding_model(query)# .merge_dims(0,1)
        doc_emb = self.embedding_model(docs).merge_dims(0,1)
        
        # reduce precision to reduce risks of similar tokens not matching to 1
        query_emb = tf.cast(query_emb, tf.float16)
        doc_emb = tf.cast(doc_emb, tf.float16)
        
        # normalize
        query_emb = tf.linalg.l2_normalize(query_emb, axis=-1)
        doc_emb = tf.linalg.l2_normalize(doc_emb, axis=-1)

        cos_similarities = tf.reduce_sum(tf.expand_dims(doc_emb, 0) * tf.expand_dims(query_emb, 1), axis=-1)
        
        bin_hashes = tf.cast((cos_similarities + 1.) / 2. * (self.bin_size-1.), tf.int32)
        
        bin_counts = tf.math.bincount(bin_hashes, minlength=self.bin_size, maxlength=self.bin_size, dtype=tf.float32, axis=-1)
        
        log_bin_counts = tf.math.log(bin_counts+1)
        
        x.update({config.MATCH_HIST_COL: log_bin_counts})
        return x