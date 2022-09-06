"""script to execute the evaluation pipeline. The pipeline consists of the following steps:
1: transforming the libsvm data into shards of tfrecords (elwc format)
2: fit a baseline ranker on a subset of the data
3: apply the ranker to generate serps
4: use a click model to simulate clicks on the serps and safe the data in a
    predefined format (elwc for pairwise, libsvm for pointwise)
5: train the model which performance is to be evaluated on the simulated data
6: check the performance (generate plots of required metrics)
"""

from mt.evaluation.simulation import settings   
from mt.data.preprocessing.libsvm_to_seq_example import write_all_shards_to_tfrecords
from mt.utils import ensure_dir_exists
from mt.evaluation.simulation.data_utils import determine_features
from mt.evaluation.simulation.fit_initial_ranker import fit_model as fit_initial_ranker
from mt.evaluation.simulation.simulate_clicks import generate_click_data
# from mt.evaluation.eval_model import main as eval_model
import os
import numpy as np


def main():
    features = determine_features(*list(settings.libsvm_path.values()))

    # ensure elwc data format is available
    if not os.path.exists(os.path.split(settings.tfr_input_data_path["train"])[0]):
        print("transform data to tfrecord")
        for kind in ["train", "test"]:
            libsvm_path = settings.libsvm_path[kind]
            output_path = settings.tfr_input_data_path[kind]
            write_all_shards_to_tfrecords(libsvm_path, features, output_path)

    # if not os.path.exists(settings.model_path):
    print("fit initial ranker")
    fit_initial_ranker(features, sample_freq=1)

    if not os.path.exists(os.path.split(settings.click_data_path["train"])[0]):
        print("generate click data")
        generate_click_data(features)

    # if not os.path.exists(settings.final_model_path):
    #     fit_final_model(features)

    # eval_model()


if __name__ == "__main__":
    main()