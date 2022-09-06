"""reading libsvm textfile using tensorflow TextLineGenerator
works only for pointwise approaches, since the assignment to queries 
is lost with this generator. An alternative is the ELWC format 
provided by tensorflow, which is esentially a dictionary of context
and examples belonging to that context.

This script tests, whether the transformation from libsvm to ELWC
format to tfrecords format and back to dense Tensors works as aspected.
"""

from mt.data.preprocessing.libsvm_to_elwc import load_shard_to_dict, write_tfrecords_shard
from mt.data.parser import ElwcParser
from collections import defaultdict
from unittest import mock
import tensorflow as tf

_PATH_TO_TEST_FILE = "./test_data/libsvm.txt"



# def test_libsvm_to_dict():
#     shard_generator = load_shard_to_dict(_PATH_TO_TEST_FILE, 1, write=False)
#     shard = next(shard_generator)

#     # test for first qid
#     examples1 = shard["1"]
#     expected_examples1 = [
#         {1: 0.123, 2: 0.001, 4: 1.0, 'label': '1'},
#         {1: 0.42, 3: 1.0, 'label': '0'},
#         {1: 0.578, 2: 1.0, 'label': '0'}
#     ]
#     pairs1 = zip(examples1, expected_examples1)
#     assert not any(x != y for x, y in pairs1)

#     # test for second qid
#     examples2 = shard["2"]
#     expected_examples2 = [
#         {1: 0.123, 2: 0.001, 4: 1.0, 'label': '1'},
#         {1: 0.42, 3: 1.0, 'label': '0'},
#         {1: 0.578, 2: 1.0, 'label': '0'}
#     ]

#     pairs2 = zip(examples2, expected_examples2)
#     assert not any(x != y for x, y in pairs2)


@mock.patch("mt.data.libsvm_to_elwc.load_shard_to_dict", lambda *_: True)
def test_libsvm_to_elwc():
    shard_generator = load_shard_to_dict(_PATH_TO_TEST_FILE, 1)
    data = next(shard_generator)
    write_tfrecords_shard(data, "test_data/test_data_*.tfrecords", 0)

    feature_map = {f"{i+1}": tf.io.FixedLenFeature([1], tf.float32, default_value=[0]) for i in range(5)}
    feature_map = {**feature_map, "label": tf.io.FixedLenFeature([1], tf.float32)}
    generator = ElwcParser(feature_map)
    dg = generator.parse_examples("test_data/test_data_0.tfrecords")
    examples = next(iter(dg))
    label = examples.pop("label")
    feat = tf.concat(list(examples.values()), -1)
    
    expected_feat = tf.constant(
        [[0.123, 0.001, 0.   , 1.   , 0.   ],
         [0.42 , 0.   , 1.   , 0.   , 0.   ],
         [0.578, 1.   , 0.   , 0.   , 0.   ]],
         dtype=tf.float32
    )
    expected_label = tf.constant(
        [[1],
         [0],
         [0]],
         dtype=tf.float32
    )

    tf.assert_equal(feat, expected_feat)
    tf.assert_equal(label, expected_label)


if __name__ == "__main__":
    pass