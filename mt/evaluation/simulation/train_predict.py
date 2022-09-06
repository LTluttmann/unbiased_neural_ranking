import tensorflow as tf
import tensorflow_ranking as tfr
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from mt.data.dataset.callbacks.feature_callbacks import LayoutEncoder
from mt.data.dataset.callbacks.feature_callbacks import FeatureMerger
from mt.data.dataset.utils import calc_label_distribution
from mt.data.dataset.parser import SeqExampleParser
from mt.evaluation.simulation import settings   
from mt.config import config
from mt.models.ultr.joe import JointEstimator
from mt.models.learning_algorithms import Metric
from mt.evaluation.simulation.scorer import SimScorer
from mt.models.learning_algorithms import SimpleTrainer
from mt.models.ultr.ipw import EMAlgorithm, IPWEstimator
from mt.evaluation.simulation.click_model import SimPosBias


layout_encoder = LayoutEncoder(None, merge_cols=["position", "layout_type"],
                               column_operation_mapping={
    # "position": lambda x: tf.one_hot(tf.cast(x, tf.int64), depth=MAX_SEQ_LEN),
    "layout_type": lambda x: tf.one_hot(tf.cast(x, tf.int64), depth=2)
})


def get_feature_map(features):
    feature_map = {
        feature_id: tf.io.FixedLenFeature([1], tf.float32, default_value=0) for feature_id in features
    }

    feature_map.update({
        config.CLICK_COL: tf.io.FixedLenFeature([1], tf.int64, default_value=-1),
        "position": tf.io.FixedLenFeature([1], tf.int64),
        "relevance": tf.io.FixedLenFeature([1], tf.int64),
        "pid": tf.io.FixedLenFeature([1], tf.string),
        "layout_type": tf.io.FixedLenFeature([1], tf.int64),
    })


def get_dataset(files, features):

    def add_merge_dim(x):
        for f in features:
            x[f] = tf.expand_dims(x[f], axis=-1)
        return x


    def copy_pos(x):
        x["actual_position"] = x["position"]
        return x

    feature_map = get_feature_map(features=features)
    context_map = {"id": tf.io.FixedLenFeature([1], tf.string)}

    parser = SeqExampleParser(feature_map, context_map, list_size=settings.max_rank)
    parser_em = SeqExampleParser(feature_map, context_map)
    
    feature_merger = FeatureMerger(features, merged_feature_name=config.NUMERIC_FEATURES_COL)

    ds = (files
          .interleave(tf.data.TFRecordDataset)
          .batch(64)
          .map(parser)
          .map(copy_pos)
          .map(add_merge_dim)
          .map(feature_merger)
          .map(layout_encoder)
          .map(lambda x: calc_label_distribution(x, [config.CLICK_COL]))
          .shuffle(10_000))

    # dataset for em algortithm
    ds_em = (files
             .interleave(tf.data.TFRecordDataset)
             .map(parser_em)
             .map(add_merge_dim)
             .map(feature_merger)
             .shuffle(10_000))
 
    return ds, ds_em


def fit_joe(ds):
    ranker = SimScorer()

    trainer = JointEstimator(ranker=ranker,
                             #losses={(f"{config.CLICK_COL}", "click_probs"): tfr.keras.losses.SigmoidCrossEntropyLoss()},
                             losses={(f"{config.CLICK_COL}_distribution",
                                      "click_probs"): tfr.keras.losses.SoftmaxLoss()},
                             metrics=[Metric("relevance", "click_probs", tfr.keras.metrics.NDCGMetric(name="ndcg1", topn=1)),
                                      Metric("relevance", "click_probs", tfr.keras.metrics.NDCGMetric(name="ndcg3", topn=3)),
                                      Metric("relevance", "click_probs", tfr.keras.metrics.NDCGMetric(name="ndcg10", topn=10))],
                             num_negatives=-1,
                             learning_rate=1e-3,
                             joe_nodes=[128, 128],
                             joe_activations="tanh",
                             joe_output_activation=None,
                             joe_multiplicative=False,
                             joe_batch_norm=False,
                             joe_input_batch_norm=True,
                             joe_dropout=0.0)

    trainer.fit(ds, epochs=3)
    return trainer


def fit_naive(ds):
    ranker = SimScorer()
    naive_trainer = SimpleTrainer(ranker, 
                                losses={f"{config.CLICK_COL}_distribution": tfr.keras.losses.SoftmaxLoss()},
                                metrics=[Metric("relevance", None, tfr.keras.metrics.NDCGMetric(name="ndcg1", topn=1)),
                                        Metric("relevance", None, tfr.keras.metrics.NDCGMetric(name="ndcg3", topn=3)),
                                        Metric("relevance", None, tfr.keras.metrics.NDCGMetric(name="ndcg10", topn=10))],
                                learning_rate = 1e-3)

    naive_trainer.fit(ds, epochs=5)

    return naive_trainer


def fit_ipw(ds, ds_em):

    em = EMAlgorithm(0.4, 0.2)
    alpha, beta = em(ds_em, 4)

    est_prop = np.array(list(beta.values()))
    est_prop /= est_prop.max()

    ranker = SimScorer()
    ipw_trainer = IPWEstimator(ranker, 
                                propensities=tf.constant(est_prop, dtype=tf.float32),
                                losses={(f"{config.CLICK_COL}_distribution", "click_scores"): tfr.keras.losses.SoftmaxLoss()},
                                metrics=[Metric("relevance", "click_scores", tfr.keras.metrics.NDCGMetric(name="ndcg1", topn=1)),
                                        Metric("relevance", "click_scores", tfr.keras.metrics.NDCGMetric(name="ndcg3", topn=3)),
                                        Metric("relevance", "click_scores", tfr.keras.metrics.NDCGMetric(name="ndcg10", topn=10))],
                                learning_rate = 1e-3)

    ipw_trainer.fit(ds, epochs=5)

    return ipw_trainer


def get_propensities_per_group(bias_estimator):

    propensities_per_group = {}



    for layout in range(settings.num_layouts):
        input_ = {
            "position": tf.expand_dims(tf.range(0, settings.max_rank, dtype=tf.int64), 0),
            "layout_type": tf.expand_dims(tf.repeat(layout, [settings.max_rank], axis=0), 0),
        }

        props = bias_estimator.predict(layout_encoder(input_)["pb_features"])
        props = props.squeeze()
        props = 1/(1+tf.math.exp(-props))
        propensities_per_group[(layout)] = props

    return propensities_per_group


def plot_propensities(joe_pb1, joe_pb2, em_pb):
    oracle=SimPosBias(label_list=list(range(4)), max_rank=settings.max_rank, control_pb_severe=[0.15, 0.6], layout_types=2)


    plt.plot(joe_pb1, label="JoE estimate for Layout A", ls="--", c='#742128', lw=1.7)
    plt.plot(joe_pb2, label="JoE estimate for Layout B", ls="--", c='#82b198', lw=1.7)
    plt.plot(em_pb, label="EM algorithm estimate", ls="--", lw=1.7, c='#bd7a98')

    plt.plot(oracle.exam_probs[0], label="Position Bias for Layout A", lw=1.7, c='#ad81b5')
    plt.plot(oracle.exam_probs[1], label="Position Bias for Layout B", lw=1.7, c='#4c72b0')

    plt.set_xlabel("position $p$")
    plt.set_ylabel("$P(E=1|p)$")
    plt.set_xticks(ticks=list(range(0, settings.max_rank, 2)), labels=list(range(1, settings.max_rank+1, 2)))

    plt.legend()
    return plt.gcf()


def main(files, features):
    ds, ds_em = get_dataset(files, features)

    ipw = fit_ipw(ds, ds_em)
    joe = fit_joe(ds)
    naive = fit_naive(ds)

    joe_pb = get_propensities_per_group(joe.propensity_estimator)
    joe_pb1 = np.array(joe_pb[0])
    joe_pb2 = np.array(joe_pb[1])

    joe_pb1 /= joe_pb1.max()
    joe_pb2 /= joe_pb2.max()

    em_pb = ipw.propensities

    fig = plot_propensities(joe_pb1, joe_pb2, em_pb)





