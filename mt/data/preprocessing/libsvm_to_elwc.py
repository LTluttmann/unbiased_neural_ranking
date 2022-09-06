from tensorflow_serving.apis import input_pb2
import tensorflow as tf
from collections import defaultdict

from mt.utils import ensure_dir_exists
from mt.data.utils import _float_feature, _bytes_feature

import os


def process_line_of_libsvm(line):
    arr = line.strip().split(' ')
    label = arr[0]
    qid = arr[1].split(":")[-1]
    features_and_label = {str(i.split(":")[0]): float(i.split(":")[-1]) for i in arr[2:]}
    features_and_label["label"] = label
    return qid, features_and_label


def load_shard_to_dict(path_to_data, shard_size):
    """this function processes large libsvm text files in shards.
    It ensures, that all events belonging to a query session fall
    in the same bucket.
    """

    fp = open(path_to_data)

    qid_features_and_label = defaultdict(list)
    counter = 0
    old_qid = ""
    for line in fp:
        qid, features_and_label = process_line_of_libsvm(line)

        # qid_features_and_label[qid].append(features_and_label)

        serp_end = False if old_qid == qid else True

        if counter >= shard_size and serp_end:
            # finished shard. Yield it and start new one
            counter = 0
            yield qid_features_and_label
            qid_features_and_label = defaultdict(list)
            qid_features_and_label[qid].append(features_and_label)
        else:
            counter += 1
            qid_features_and_label[qid].append(features_and_label)

        old_qid = qid
    # also return unfull shards
    yield qid_features_and_label
    fp.close()


def write_tfrecords_shard(data, path, shard_num):
    with tf.io.TFRecordWriter(path.replace("*", str(shard_num))) as writer:
        for i, (qid, examples) in enumerate(data.items()):
            context_dict = {}
            context_dict["id"] = _bytes_feature([str.encode(qid)])
            context_proto = tf.train.Example(features=tf.train.Features(feature=context_dict))

            ELWC = input_pb2.ExampleListWithContext()
            ELWC.context.CopyFrom(context_proto)

            for example in examples:
                example_features = ELWC.examples.add()

                example_dict = {
                    feature_id: _float_feature([float(feature_val)])
                    for feature_id, feature_val in example.items()}
                exampe_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
                example_features.CopyFrom(exampe_proto)

            writer.write(ELWC.SerializeToString())


def write_all_shards_to_tfrecords(libsvm_path, output_path):
    ensure_dir_exists(os.path.split(output_path)[0])
    shard_generator = load_shard_to_dict(libsvm_path, 25000)
    for i, shard in enumerate(shard_generator):
        print("write shard as tfrecord...")
        write_tfrecords_shard(shard, output_path, i)
