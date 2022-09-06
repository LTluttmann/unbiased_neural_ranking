import tensorflow as tf

from mt.config import config


def infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


@tf.function
def merge_first_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[0] *= shape[1]
    merge_dim = shape.pop(1)
    return tf.reshape(tensor, shape), merge_dim


def merge(x):
    for k,v in x.items():
        if k ==config.QUERY_COL:
            continue
        x[k], merge_dim = merge_first_two_dims(v)
    x[config.QUERY_COL] = tf.repeat(x[config.QUERY_COL], [merge_dim], axis=0)
    return x
    

@tf.function
def has_positives(x):
    """filter static lists (only clicks or no clicks at all)
    NOTE: filtering is expensive and should be done with bigquery / spark. 
    Leave it here anyways
    """
    has_positives = tf.math.reduce_any(tf.equal(x[config.CLICK_COL], 1), axis=-1)
    return has_positives

@tf.function
def has_positives_and_negatives(x):
    """filter static lists (only clicks or no clicks at all)
    NOTE: filtering is expensive and should be done with bigquery / spark. 
    Leave it here anyways
    """
    has_positives = tf.math.reduce_any(tf.equal(x[config.CLICK_COL], 1), axis=-1)
    has_negatives = tf.math.reduce_any(tf.equal(x[config.CLICK_COL], 0), axis=-1)
    return tf.math.logical_and(has_positives, has_negatives)


@tf.function
def concat_from_zipped_datasets(a,b):
    """concatenates the second dimension from two datasets"""
    a = a.copy()
    query = a.pop(config.QUERY_COL)
    batch_size = tf.shape(query)[0]

    for k,v in a.items():
        # define the default values here!!!
        if k in config.QUERY_ITEM_FEATURES + config.POSITION_BIAS_FEATURES + config.LABELS:
            random_negatives = tf.zeros_like(tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0))
        elif k == config.SAMPLING_WEIGHT_COL:
            random_negatives = tf.ones_like(tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0))
        else:
            random_negatives = tf.repeat(tf.transpose(b[k], perm=[1,0]), [batch_size], axis=0)
        a[k] = tf.concat((v, random_negatives), axis=1)
    a[config.QUERY_COL] = query
    return a


class SampleWeighter():

    # WEIGHT_FUNCTION = {
    #     "click": lambda x: tf.maximum(1.0, tf.cast(x[config.CLICK_COL], tf.float32) * 0.0),
    #     "ordered": lambda x: tf.maximum(1.0, tf.cast(x[config.ORDER_COL], tf.float32) * (500.0 + (x[config.PRICE_COL]) / 1.0) )
    # }

    def __init__(self, num_tasks, weight_function=None) -> None:
        self.num_tasks = num_tasks
        self.weight_function = weight_function or {}

    def _add_sample_weight(self, x):
        w = tf.maximum(1.0, tf.cast(x[config.ORDER_COL], tf.float32) * (x[config.PRICE_COL] / 3.5))
        x[config.SAMPLE_WEIGHT_ON_LOSS_COL] = tf.cast(w, tf.float32)
        return x

    def _add_sample_weight_per_output(self, x):
        weights = {}
        for col in config.LABELS:
            weight_fn = self.weight_function.get(col, None)
            if weight_fn:
                weights[col] = tf.cast(weight_fn(x), tf.float32)
            else:
                weights[col] = tf.ones_like(x[col], tf.float32)
        x[config.SAMPLE_WEIGHT_ON_LOSS_COL] = weights
        return x

    def __call__(self, x):
        if self.num_tasks > 1:
            return self._add_sample_weight_per_output(x)
        else:
            return self._add_sample_weight(x)


def calc_label_distribution(x, label_cols=config.LABELS):
    for col in label_cols:
        # [BS, S]
        y = x[col]

        mask = tf.not_equal(y, -1)
        label_sum = tf.reduce_sum(tf.ragged.boolean_mask(y, mask), axis=1, keepdims=True)

        P_y = tf.math.divide_no_nan(tf.cast(y, tf.float32), tf.cast(label_sum, tf.float32))
        x[f"{col}_distribution"] = tf.where(mask, P_y, tf.cast(y, tf.float32))
    return x

# def pbk_match(x):
#     # NOTE: deprecated, use layers instead
#     #q = tokenizer.tokenize(a["searchterm"]).merge_dims(-2,-1)
#     #q = token_ops.pad_sequence_new(q, 32, a["searchterm"].get_shape().as_list()+[32])
#     q = x["searchterm"]
    
#     class_pred = pbk_classifier(q, training=False)
#     class_pred = class_pred[:, tf.newaxis, :]
#     pbk_one_hot = tf.one_hot(pbk_lookup(x["product_pd_AttributePbk"]), depth=len(pbks)+1)
#     class_feature = tf.reduce_sum(pbk_one_hot * class_pred, axis=-1)
#     x["pbk_match"] = tf.where(tf.equal(x["click"],-1), -1.0, class_feature)
#     return x

# def calc_semantic_matching_score(x):
#     # NOTE: deprecated, use layers instead
#     bs = tf.shape(x["product_pd_Name"])[0]
#     seq_len = tf.shape(x["product_pd_Name"])[1]
#     token_len = tf.shape(x["product_pd_Name"])[2]
    
#     q_emb = encoder(x["searchterm"], training=False)
#     emb_dim = tf.shape(q_emb)[1]
    
#     doc_emb = encoder(tf.reshape(x["product_pd_Name"], (-1, token_len)), training=False)
#     doc_emb = tf.reshape(doc_emb, (bs, seq_len, emb_dim))
#     sms = tf.keras.layers.Dot(axes=[1,2], normalize=True)([q_emb, doc_emb])
#     sms = tf.where(tf.math.is_nan(sms), tf.zeros_like(sms), sms)
#     x["sms"] = tf.where(tf.equal(x["click"],-1), -1.0, sms)
#     return x