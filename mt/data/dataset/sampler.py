import tensorflow as tf

from mt.data.dataset.dataset_ops import BaseDatasetOp
from mt.config import config 


class SequenceExampleSampler(BaseDatasetOp):

    def __init__(self, num_negatives:int = 1, replacement=True, sample_weight=None, on_batch: bool = True) -> None:
        super().__init__(on_batch)
        self.num_negatives = num_negatives
        self.sample_weight = sample_weight
        self.neg_sample_fn = self.sample_w_replacement if replacement else self.sample_without_replacement


    def sample_without_replacement(self, logits):
        """
        use the gumbel-max trick for categotical sampling from logits
        Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
        """
        # logits = tf.math.log(odds / (1-odds))
        # we need the log on the unnormalized probabilities in order to make sure, items with probabilitiy of zero are never chosen
        logits = tf.math.log(logits)
        z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits),0,1)))
        _, indices = tf.nn.top_k(logits + z, self.num_negatives)
        return tf.cast(indices, tf.int64)

    def sample_w_replacement(self, logits):
        return tf.random.categorical(tf.math.log(logits), self.num_negatives)


    @tf.function
    def call_on_batch(self, example: dict):

        """For a binary label (usually 0 and 1, but -1, 1 could also happen) this
        function samples one example per target value

        Args:
            example (dict): _description_
            label (str): _description_
            pos_label (int, optional): _description_. Defaults to 1.
            neg_label (int, optional): _description_. Defaults to 0.
        """
        if self.num_negatives == -1:
            example = example.copy()
            _ = example.pop(self.sample_weight, None)
            return example
        else:
            example = example.copy()

            # [BS, None] -> SERP length is the ragged dimension
            y = example.get(config.CLICK_COL)
            y_o = example.get(config.ORDER_COL, tf.zeros_like(y))
            # Ragged: [BS, None] -> SERP length is the ragged dimension
            query = example.pop(config.QUERY_COL)

            probs_neg = tf.cast(example.pop(self.sample_weight, tf.equal(y, 0)), tf.float32)
            # make sure we have zero probability mass on clicked items. However, allow for padded items with very low prob
            probs_neg = tf.where(tf.equal(y, 1), 0.0, probs_neg)
            probs_neg = tf.where(tf.equal(y, -1), 1e-9, probs_neg)

            idx_neg = self.neg_sample_fn(probs_neg)
            #neg_label = tf.zeros_like(idx_neg)

            probs_pos = tf.cast(tf.equal(y, 1), tf.float32) + tf.cast(tf.equal(y_o, 1), tf.float32)
            idx_pos = tf.random.categorical(tf.math.log(probs_pos), 1)
            #pos_label = tf.ones_like(idx_pos)

            stacked_examples = {}
            for k, t in example.items():
                
                # return t, idx_pos
                pos = tf.gather(t, idx_pos, batch_dims=1, axis=1)
                neg = tf.gather(t, idx_neg, batch_dims=1, axis=1)

                stacked_examples[k] = tf.concat((pos, neg), axis=1)

            stacked_examples[config.QUERY_COL] = query
            #stacked_examples[config.LABEL_COL] = tf.concat((pos_label, neg_label), axis=1)

            return stacked_examples

    @tf.function
    def call_on_single_example(self):
        """For a binary label (usually 0 and 1, but -1, 1 could also happen) this
        function samples one example per target value

        Args:
            example (dict): _description_
            label (str): _description_
            pos_label (int, optional): _description_. Defaults to 1.
            neg_label (int, optional): _description_. Defaults to 0.
        """
        example = example.copy()
        # pop out label, it is not relevant. # TODO check if passing labels might be more intuitive
        # [BS, 1]
        y = example.pop(config.CLICK_COL)
        # Ragged: [BS, None]
        query = example.pop(config.QUERY_COL)

        probs_neg = tf.expand_dims(tf.cast(tf.equal(y, 0), tf.float32), 0)
        idx_neg = tf.random.categorical(tf.math.log(probs_neg), self.num_negatives)
        neg_label = tf.zeros((self.num_negatives,))

        probs_pos = tf.expand_dims(tf.cast(tf.equal(y, 1), tf.float32), 0)
        idx_pos = tf.random.categorical(tf.math.log(probs_pos), 1)
        pos_label = tf.zeros((1,))

        stacked_examples = {}
        for k, t in example.items():
            pos = tf.gather(t, idx_pos)
            neg = tf.gather(t, idx_neg)
            
            if isinstance(pos, tf.RaggedTensor):
                pos = pos.merge_dims(-2,-1)
                neg = neg.merge_dims(-2,-1)
                
            pos = tf.reshape(pos, (1, -1))
            neg = tf.reshape(neg, (self.num_negatives, -1))

            stacked_examples[k] = tf.concat((pos, neg), axis=0)

        stacked_examples[config.QUERY_COL] = query
        stacked_examples[config.CLICK_COL] = tf.concat((pos_label, neg_label), axis=1)
        
        return stacked_examples


class EasyNegativeSampler(BaseDatasetOp):

    def __init__(self, num_negatives:int = 1, on_batch: bool = True) -> None:
        super().__init__(on_batch)
        self.num_negatives = num_negatives

    @tf.function
    def _get_negative_indices(self, bs):
        # first generate a matrix with all indices
        # [[0,1,2,...], 
        # [0,1,2,...]
        # [0,1,2,...]]
        include_all_indeces = tf.ones((bs,1), dtype=tf.int32) * tf.range(0,bs)
        # generate a boolean mask where all negatives are True
        boolean_mask = tf.logical_not(tf.cast(tf.eye(bs), tf.bool))
        # sample indices according to boolean mask 
        indices = tf.reshape(tf.boolean_mask(include_all_indeces, boolean_mask), (bs, bs-1))
        return indices

    @tf.function
    def call_on_batch(self, example: dict):
        example = example.copy()
        # batch size is number of negatives + 1 (for the positive)
        bs = self.num_negatives + 1
        negative_indices = self._get_negative_indices(bs)
        # we need to pop the query col, as the context is handled differently (no "example dimension")
        stacked_examples = {config.QUERY_COL: example.pop(config.QUERY_COL)}

        for k, t in example.items():
            pos = t[:,tf.newaxis,:]
            neg = tf.gather(t, negative_indices)

            stacked_examples[k] = tf.concat((pos, neg), axis=1)
            
        # define a new label
        pos_label = tf.ones((bs, 1, 1))
        neg_label = tf.zeros((bs, self.num_negatives, 1))
        stacked_examples[config.CLICK_COL] = tf.concat((pos_label, neg_label), axis=1)

        return stacked_examples

    
    @tf.function
    def call_on_single_example(self, example):
        return self.call_on_batch(example)
