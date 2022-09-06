# import nn stuff
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras as nn

# import own modules
from mt.utils import make_or_delete_dir, ensure_dir_exists,ensure_dirs_exists
from mt.data.dataset.parser import SeqExampleParser
from mt.data.utils import _bytes_feature, _float_feature, _int64_feature
from mt.evaluation.simulation.click_model import SimPosBias, SimClickAndPurchase, ClickModel
from mt.evaluation.simulation import settings
from mt.config import config
# import misc
from dataclasses import dataclass
from typing import List
import numpy as np
from typing import Union
import os


@dataclass
class Serp:
    qid: str
    relevances: Union[List[int], np.ndarray]
    features: Union[List[float], np.ndarray]
    feature_keys: List
    pid: List
    clicks: List[int] = None
    layout_type:int = None

    def write_line_in_libsvm_format(self, idx):
        click = self.clicks[idx]
        feat_str = " ".join([f"{i}:{val}" for i, val in enumerate(self.features[idx]) if val!=0])
        
        return f"{click} qid:{self.qid} pid:{self.pid[idx]} pos:{idx} {feat_str} \n"
        
    def write_serp_in_libsvm_format(self):
        return "".join([self.write_line_in_libsvm_format(i) for i in range(len(self.relevances))])


    def write_serp_in_tfrecord_format(self):
        context_dict = {}
        context_dict["id"] = _bytes_feature(self.qid)
        context_proto = tf.train.Features(feature=context_dict)

        features_lists = {}
        for i, X in enumerate(self.features.T):
            features_list = []
            for x in X:
                features_list.append(_float_feature([x]))
            feature_key = self.feature_keys[i]
            features_lists[str(feature_key)] = tf.train.FeatureList(feature=features_list)

        relevance_list = []
        click_list = []
        position_list = []
        pid_list  = []
        layout_list = []
        for pos, (rel, click, pid) in enumerate(zip(self.relevances, self.clicks, self.pid)):
            relevance_list.append(_int64_feature([int(rel)]))
            click_list.append(_int64_feature([int(click)]))
            position_list.append(_int64_feature([int(pos)]))
            pid_list.append(_bytes_feature([pid]))
            if self.layout_type is not None:
                layout_list.append(_int64_feature([int(self.layout_type)]))

        features_lists["relevance"] = tf.train.FeatureList(feature=relevance_list)
        features_lists["position"] = tf.train.FeatureList(feature=position_list)
        features_lists[config.CLICK_COL] = tf.train.FeatureList(feature=click_list)
        features_lists["pid"] = tf.train.FeatureList(feature=pid_list)
        if self.layout_type is not None:
            features_lists["layout_type"] = tf.train.FeatureList(feature=layout_list)


        seq_ex = tf.train.SequenceExample(
            context = context_proto,
            feature_lists = tf.train.FeatureLists(feature_list=features_lists)
        )

        return seq_ex.SerializeToString()


def generate_serp(new_qid, ranker, X, y, max_rank, feature_keys, pid, std, layout_type) -> Serp:
    X, y = X.numpy(), y.numpy()
    y = y.astype("int")
    pid = pid.numpy()

    score = ranker.predict(X)
    score = score.squeeze(-1)
    score += np.random.normal(scale=std, size=score.shape)
    # sort pids according to rank. argsort performs in ascending order, thus take negative of score
    sorted_idx = (-score).argsort()

    X = X[sorted_idx][:max_rank]
    y = y[sorted_idx][:max_rank]
    # generate a serp
    serp = Serp(new_qid, y, X, feature_keys, pid, layout_type=layout_type)
    return serp


def simulate_and_write_data(
    dataset: tf.data.Dataset,
    ranker: nn.Model,
    click_model: ClickModel, 
    feature_keys: list,
    kind: str = "train",
    std:float = 0.0,
    writer=tf.io.TFRecordWriter
):
    feature_keys = sorted([int(x) for x in feature_keys])
    shard, step = 0, 0
    dataset = iter(dataset)
    while step < settings.num_samples[kind]:
        file_path = settings.click_data_path[kind].replace("*", str(shard))
        output_format = file_path.split(".")[-1]
        with writer(file_path) as w:
            for _ in range(settings.shard_size):
                # we dont want the same qid to appear multiple times
                if settings.num_layouts > 1:
                    observed_layout_type = int(round(np.random.random(),0))
                else:
                    observed_layout_type = None

                X, y, qid, pid = next(dataset)
                # generate a serp
                serp = generate_serp(qid, ranker, X, y, settings.max_rank, feature_keys, pid, std, observed_layout_type)
                # simulate clicks for the serp
                serp.clicks = click_model.sim_clicks_for_serp(serp)
                # write serp to file
                if output_format == "tfrecords":
                    output_string = serp.write_serp_in_tfrecord_format()
                elif output_format == "txt":
                    output_string = serp.write_serp_in_libsvm_format()
                else:
                    raise NotImplementedError(f"output format must be one of txt ot tfrecords, got {output_format}")
                w.write(output_string)

                step += 1
                if step % 500 == 0: print(step)

        shard += 1


def get_dataset(file_path, feature_list):
    def parse_fn(examples:dict):

        qid = examples.pop("id")
        label = examples.pop("label")
        sorted_dict = dict(sorted({int(k): tf.expand_dims(v,-1) for k,v in examples.items()}.items()))
        feat = tf.concat(list(sorted_dict.values()), -1)

        a=tf.repeat("qid"+qid+"_", [tf.shape(feat)[0]], axis=0)
        b=tf.strings.as_string(tf.range(1,tf.shape(feat)[0]+1))
        pid = tf.strings.join([a,b])

        return feat, label, qid, pid

    feature_map = {
        feature_id: tf.io.FixedLenFeature([1], tf.float32, default_value=[0]) 
        for feature_id in feature_list}
    feature_map = {**feature_map, "label": tf.io.FixedLenFeature([], tf.float32)}
    context_map = {"id": tf.io.FixedLenFeature([1], tf.string)}
    parser = SeqExampleParser(feature_map, context_map)
    dataset = (
        tf.data.Dataset
        .list_files(file_path, shuffle=False) # .take(7)
        .interleave(tf.data.TFRecordDataset)
        .map(parser, tf.data.AUTOTUNE)
        .map(parse_fn, tf.data.AUTOTUNE)
        .shuffle(10_000)
        .take(2_000)
    )
    return dataset


def generate_click_data(feature_list, std=0.0) -> None:

    # load data and weak ranker
    model = nn.models.load_model(settings.model_path)
    ranker = model.mlp
    nu = 0.5 if settings.num_layouts == 1 else [0.15, 0.6]
    click_model = SimPosBias(list(range(5)), settings.max_rank, click_noise=0.1, control_pb_severe=nu, layout_types=settings.num_layouts)
    # create directory if not exists
    ensure_dirs_exists(os.path.split(settings.click_data_path["train"])[0])
    for kind in ["train", "test"]:
        # simulate data for given dataset
        dataset = get_dataset(settings.tfr_input_data_path[kind], feature_list) # .cache()
        dataset = (
            dataset
            .shuffle(10_000)
            .repeat(-1)
        )
        simulate_and_write_data(dataset, ranker, click_model, feature_list, kind=kind, std=std)


if __name__ == "__main__":
    from mt.evaluation.simulation.data_utils import determine_features
    features = determine_features(*list(settings.libsvm_path.values()))


    stds = np.linspace(0,1,num=6)
    stds = [0.2]
    
    if stds is not None: 
        orig_path = settings.click_data_path["train"]
        for std in stds:

            split_path =  os.path.split(orig_path)
            root, file = split_path
            settings.click_data_path["train"] = os.path.join(root, str(std).replace(".", ","), file)

            generate_click_data(features, std=std)
    else:

        generate_click_data(features)
