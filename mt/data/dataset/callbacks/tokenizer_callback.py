import tensorflow as tf

from mt.data.dataset.dataset_ops import BaseDatasetOp


class TokenizerCallback(BaseDatasetOp):

    def __init__(self, tokenizer, cols, max_length=None, batch_processing=True) -> None:
        super().__init__(on_batch=batch_processing)
        self.tokenizer = tokenizer
        self.token_cols = cols
        self.token_length = max_length if max_length else {}

    @tf.function
    def pad_sequence_new(self, t, maxlen, static_shape, pad_val=0):

        def pad_fn():
            pad_vals = tf.ones(tf.maximum(0, maxlen-tf.shape(t)[-1]), dtype=t.dtype) * pad_val
            return tf.concat((t, pad_vals), axis=-1)

        def truncate_fn():
            return t[..., :maxlen]
        
        if isinstance(t, tf.RaggedTensor):
            t = t.to_tensor(shape=static_shape)
        else:
            t = tf.cond(tf.shape(t)[-1] >= maxlen, truncate_fn, pad_fn)

        t.set_shape(shape=static_shape)
        
        return t

    def call_on_single_example(self, inputs: dict):
        self.call_on_batch(inputs)
    
    def call_on_batch(self, inputs):
        x = inputs.copy()
        for col in self.token_cols:
            t = x[col]
            static_shape = t.get_shape().as_list()
            
            if len(static_shape) > 0:
                tokens = self.tokenizer.tokenize(t).merge_dims(-2, -1)
            else:
                tokens = self.tokenizer.tokenize(t).merge_dims(-2, -1).merge_dims(-2, -1)

            maxlen =self.token_length.get(col, None)

            if maxlen:
                static_shape.append(maxlen)
                tokens = self.pad_sequence_new(tokens, maxlen, static_shape)

            x[col] = tokens
            
        return x