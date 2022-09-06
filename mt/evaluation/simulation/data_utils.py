import os
from collections import defaultdict
import pandas as pd
import tensorflow as tf
import numpy as np


def determine_features(train_path, test_path):
    features = set()
    for file in [train_path, test_path]:
        fp = open(file)
        for line in fp:
            array = line.split(" ")
            feat = [i.split(":")[0] for i in array[2:]]
            features = features.union(set(feat))
    features: np.array = np.array(list(features)).astype("int")
    features.sort()
    return features.astype("str").tolist()


def sample_pair_from_relative_preferences(X, y):
    y = tf.squeeze(y, -1)
    random_sample = tf.random.categorical(tf.math.log(tf.ones_like(tf.expand_dims(y, 0))), 1)
    random_sample = tf.squeeze(random_sample)
    X1, y1 = tf.gather(X, random_sample), tf.gather(y, random_sample)

    probs = tf.expand_dims(tf.cast(tf.not_equal(y, y1), tf.float32), 0)
    idx = tf.random.categorical(tf.math.log(tf.reshape(probs, [1, -1])), 1)
    idx = tf.squeeze(idx)
    # slice observations using index
    X2, y2 = tf.gather(X, idx, axis=0), tf.gather(y, idx)

    # Uniform variable in [0,1)
    p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)

    def return_positive_first():
        new_label = tf.constant([1])
        return tf.cond(
            tf.greater(y1, y2), 
            lambda: ((X1, X2), new_label), 
            lambda: ((X2, X1), new_label)
        )

    def return_negative_first():
        new_label = tf.constant([0])
        return tf.cond(
            tf.less(y1, y2), 
            lambda: ((X1, X2), new_label), 
            lambda: ((X2, X1), new_label)
        )

    return tf.cond(pred, return_positive_first, return_negative_first)


def process_line(line):
    arr = line.strip().split(' ')
    label = arr[0]
    qid = arr[1].split(":")[-1]
    features_and_label = {i.split(":")[0]: i.split(":")[-1] for i in arr[2:]}
    features_and_label["label"] = label
    return qid, features_and_label


def load_data_to_df(path_to_data, limit:int=float("inf")):
    qid_features_and_label = {}
    qid_docs = defaultdict(list)
    with open(os.path.join(path_to_data), "r") as f:
        a = f.readlines()
        for i, line in enumerate(a):
            qid, features_and_label = process_line(line)
            len_q = 0 if not qid_docs.get(qid, None) else len(qid_docs[qid])

            pid = f"q{qid}p{len_q+1}"
            qid_docs[qid].append(pid)
            
            qid_features_and_label[(qid,pid)] = features_and_label
            if (i+1) % 10000 == 0:
                print(i)
            if i >= limit:
                break

    # missing features correspond to them being zero values
    df_labeled = pd.DataFrame.from_dict(qid_features_and_label, orient="index").fillna(0)
    df_labeled.index.names = ["qid", "pid"]
    return df_labeled

def save_as_txt_file():
    pass
