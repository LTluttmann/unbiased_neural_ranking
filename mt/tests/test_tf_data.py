import tensorflow as tf
from unittest import mock

from mt.data.preprocessing.libsvm_to_elwc import write_all_shards_to_tfrecords
from mt.data.generators import PairGenerator
from mt.data.parser import ElwcParser
from mt.evaluation.simulate_clicks import simulate_and_write_data, Serp, _get_dataset
from mt.data.preprocessing.libsvm_to_elwc import load_shard_to_dict, write_tfrecords_shard


def random_number():
    return tf.random.uniform([], 0 , 1)


def test_sampling():
    label = tf.constant([[1.0, 2.0, 2.0, 1.0, 2.0]])
    # probabilities of obs with label < 2 is set to zero here
    pos_probs = tf.cast(tf.greater(label, 1), tf.float32)
    # determine indices of obs with label greater 1
    pos_candidates = tf.where(tf.greater(label, 1))
    # probabilities of obs with label > 1 is set to zero here
    neg_probs = tf.cast(tf.less(label, 2), tf.float32)
    # determine indices of obs with label greater 1
    neg_candidates = tf.where(tf.less(label, 2))
    # make sure no idx has nonzero prob for both, pos and neg
    assert ~tf.reduce_any(tf.greater(tf.add(pos_probs, neg_probs), 1))
    # do serveral runs
    for _ in range(100):
        # check if positive sample is in candidate list
        pos_idx = tf.random.categorical(tf.math.log(pos_probs), 1)
        assert tf.reduce_any(tf.equal(pos_idx, pos_candidates))
        # check if negative sample is in candidate list
        neg_idx = tf.random.categorical(tf.math.log(neg_probs), 1)
        assert tf.reduce_any(tf.equal(neg_idx, neg_candidates))


@mock.patch("tensorflow.random.uniform")
def test_random_number(uniform):
     
    mock1 = lambda *_: 1
    mock2 = lambda *_: 2

    uniform.side_effect = [mock1(), mock2()]

    assert random_number() == 1
    assert random_number() == 2
    

@mock.patch("tensorflow.random.categorical")
@mock.patch("tensorflow.random.uniform")
# mocks go as function arguments in REVERSED order!
def test_sample_pair_from_relative_preferences(random_uniform, random_categorical):
    # mock results of random number generators
    random_categorical.side_effect = [tf.constant([0]), tf.constant([1])]
    random_uniform.side_effect = [0.0]

    y = tf.constant([[1.0], [0.0]])
    X = tf.constant([[0.2, 0.4], [0.1, 0.8]])
    pos, neg, label = PairGenerator.sample_pair_from_relative_preferences(X, y)

    expected_pos = tf.constant([[[0.2, 0.4]]])
    expected_neg = tf.constant([[[0.1, 0.8]]])
    expected_label = tf.constant([1])

    tf.assert_equal(pos, expected_pos)
    tf.assert_equal(neg, expected_neg)
    tf.assert_equal(label, expected_label)



@mock.patch("mt.evaluation.simulate_clicks.generate_serp")
@mock.patch("mt.evaluation.settings.num_samples", {"train": 1})
@mock.patch("mt.evaluation.settings.max_rank", 20)
@mock.patch("mt.evaluation.settings.shard_size", 1)
@mock.patch("mt.evaluation.settings.click_data_path", {"train": "test_data/test_click_data_*.tfrecords"})
def test_generate_click_data(serp):
    def _parse_fn(examples:dict):
        label = examples.pop("label")
        feat = tf.concat(list(examples.values()), -1)
        return feat, label

    shard_generator = load_shard_to_dict("test_data/libsvm.txt", 1)
    data = next(shard_generator)
    write_tfrecords_shard(data, "test_data/test_data_*.tfrecords", 0)
    
    feature_map = {f"{i+1}": tf.io.FixedLenFeature([1], tf.float32, default_value=[0]) for i in range(5)}
    feature_map = {**feature_map, "label": tf.io.FixedLenFeature([1], tf.float32)}
    parser = ElwcParser(feature_map)
    dg = parser.parse_examples("test_data/test_data_0.tfrecords").map(lambda example: _parse_fn(example))
    feat, label = next(iter(dg))
    
    mock_serp = Serp("0", label.numpy().squeeze(-1), feat.numpy())
    serp.side_effect = [mock_serp]

    CLICK_LIST = [1,0,0]
    class MockClickModel:
        def sim_clicks_for_serp(*args):
            return CLICK_LIST
    
    
    dg = parser.parse_examples("test_data/test_data_0.tfrecords").map(lambda example: _parse_fn(example))
    simulate_and_write_data(dg, None, MockClickModel)


    feature_map = {
        f"{i+1}": tf.io.FixedLenFeature([1], tf.float32, default_value=[0]) for i in range(5)
    }
    feature_map.update({
        "label": tf.io.FixedLenFeature([1], tf.int64),
        "position": tf.io.FixedLenFeature([1], tf.int64),
        "relevance": tf.io.FixedLenFeature([1], tf.int64)
    })
    parser = ElwcParser(feature_map)
    dg = parser.parse_examples("test_data/test_click_data_*.tfrecords")
    example = next(iter(dg))

    label = example.pop("label")
    relevance = example.pop("relevance")
    position = example.pop("position")
    position_vec = tf.one_hot(tf.reshape(position, shape=(-1,)), depth=20)
    feat = tf.concat(list(example.values()), axis=1)

    expected_label = tf.reshape(tf.constant(CLICK_LIST, dtype=tf.int64), (-1,1))
    expected_feat = tf.constant(
        [[0.123, 0.001, 0., 1., 0.],
         [0.42, 0., 1., 0., 0.],
         [0.578, 1., 0., 0., 0.]], dtype=tf.float32
    )
    expected_position_vec = tf.gather(tf.eye(20), tf.constant([0, 1, 2]))
    expected_relevance = tf.reshape(tf.constant([1, 0, 0], dtype=tf.int64), (-1,1))

    tf.assert_equal(label, expected_label)
    tf.assert_equal(position_vec, expected_position_vec)
    tf.assert_equal(feat, expected_feat)
    tf.assert_equal(relevance, expected_relevance)
