from dataclasses import dataclass, asdict
import os
import json
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# define constants
MAX_RANK = 15
SAMPLES_TRAIN = 0
SAMPLES_TEST = 5_000
NUM_QIDS_IN_SHARD = 1000
NUM_LAYOUTS = 2

# root folder path
EVAL_RESULTS_ROOT_PATH = f"results"

# path to yahoo letor dataset files
LIBSVM_PATH_TRAIN = "/Users/Laurin/Downloads/Learning to Rank Challenge/ltrc_yahoo/set1.train.txt"
LIBSVM_PATH_TEST = "/Users/Laurin/Downloads/Learning to Rank Challenge/ltrc_yahoo/set1.valid.txt"

# path to elwc files
LETOR_TFR_DATA_TRAIN_PATH = os.path.join("/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/letor2", "data_train_*.tfrecords")
LETOR_TFR_DATA_TEST_PATH = os.path.join("/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/letor2", "data_test_*.tfrecords")

# initial ranker training files
INIT_RANKER_TRAIN_PATH = LETOR_TFR_DATA_TRAIN_PATH
INIT_RANKER_TEST_PATH = LETOR_TFR_DATA_TEST_PATH

# path to data with simulated clicks
LAYOUT_PREFIX = "" if NUM_LAYOUTS == 1 else "_multilayout"
CLICK_DATA_TRAIN_PATH = os.path.join(f"/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/sim_data{LAYOUT_PREFIX}", "data_train_*.tfrecords")
CLICK_DATA_TEST_PATH = os.path.join(f"/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/sim_data{LAYOUT_PREFIX}", "data_test_*.tfrecords")

MODEL_PATH = "/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/weak_ranker"
FINAL_MODEL_PATH = "/Users/Laurin/Documents/Uni/Masterarbeit/jarvis_mt-marvel/final_model"


# put constants in a container class
@dataclass
class SettingsContainer:
    max_rank: int
    num_samples: dict
    shard_size: int
    root_path: str
    click_data_path: dict
    libsvm_path: dict
    tfr_input_data_path: dict
    init_ranker_input_path: dict
    model_path: str
    final_model_path: str
    num_features: int=None
    num_layouts: int=1
    
    def to_json(self):
        with open(os.path.join(self.root_path, 'meta_data.json'), 'w') as fp:
            json.dump(asdict(self), fp)


settings = SettingsContainer(
    max_rank=MAX_RANK, 
    num_samples={"train": SAMPLES_TRAIN, "test": SAMPLES_TEST},
    shard_size=NUM_QIDS_IN_SHARD,
    click_data_path={"train": CLICK_DATA_TRAIN_PATH, "test": CLICK_DATA_TEST_PATH},
    libsvm_path={"train": LIBSVM_PATH_TRAIN, "test": LIBSVM_PATH_TEST},
    tfr_input_data_path={"train": LETOR_TFR_DATA_TRAIN_PATH, "test": LETOR_TFR_DATA_TEST_PATH},
    root_path=EVAL_RESULTS_ROOT_PATH,
    model_path=MODEL_PATH,
    final_model_path=FINAL_MODEL_PATH,
    init_ranker_input_path={"train": INIT_RANKER_TRAIN_PATH, "test": INIT_RANKER_TEST_PATH},
    num_layouts=NUM_LAYOUTS
)


def get_settings_from_dict(path):
    with open(path, 'r') as fp:
        set_dict = json.load(fp)
    
    return SettingsContainer(**set_dict)
