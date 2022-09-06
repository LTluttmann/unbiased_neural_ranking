import tensorflow as tf
import tensorflow.keras as nn

from mt.data.dataset.parser import SeqExampleParser
from mt.utils import ensure_dir_exists
from mt.evaluation.simulation import settings
from mt.models.callbacks import ErrorLogsCallback
from mt.evaluation.simulation.data_utils import sample_pair_from_relative_preferences
from mt.models.layers import create_tower

from typing import List
import matplotlib.pyplot as plt


_EPOCHS = 20
_BATCH_SIZE = 64

loss_callback = ErrorLogsCallback(_EPOCHS)

lr_reducer = nn.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
)


early_stopping = nn.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=10,
)

save_best_model = nn.callbacks.ModelCheckpoint(
    settings.model_path,
    monitor="val_loss",
    verbose=10,
    save_best_only=True,
    save_weights_only=False,
)

def plot_metrics(train_metric, val_metric=None, metric_name=None, title=None, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(train_metric,color='blue',label=metric_name)
    if val_metric is not None: plt.plot(val_metric,color='green',label='val_' + metric_name)
    plt.legend(loc="upper right")
    plt.show()

# model architecture
class RankNet(nn.Model):
    def __init__(self):
        super().__init__()
        self.mlp = create_tower([512, 512], 1, activation="relu", dropout=0.5, use_batch_norm=True)
        self.oi_minus_oj = nn.layers.Subtract()
    
    def call(self, inputs, training=True):
        xi = inputs[0]
        xj = inputs[1]
        oi = self.mlp(xi, training=training)
        oj = self.mlp(xj, training=training)
        oij = self.oi_minus_oj([oi, oj])
        output = nn.layers.Activation('sigmoid')(oij)
        return output


def fit_model(features: List[str], batch_size=128, sample_freq=1):

    def parse_fn(examples:dict):
        """concat features and extract label"""
        label = examples.pop("label")
        sorted_dict = dict(sorted({int(k): v for k,v in examples.items()}.items()))
        feat = tf.concat(list(sorted_dict.values()), -1)
        return feat, label

    def get_pairs_generator(file_pattern) -> tf.data.Dataset:
        feature_map = {feature_id: tf.io.FixedLenFeature([1], tf.float32, default_value=[0]) for feature_id in features}
        feature_map = {**feature_map, "label": tf.io.FixedLenFeature([1], tf.float32)}
        parser = SeqExampleParser(feature_map)
        files = tf.data.Dataset.list_files(file_pattern, shuffle=False)#.take(5).cache()
        ds = (
            files
            .interleave(tf.data.TFRecordDataset)
            .map(parser, tf.data.AUTOTUNE)
            .map(parse_fn, tf.data.AUTOTUNE)
            # remove constant labeled batches
            .filter(lambda X, y: tf.reduce_any(tf.not_equal(y, y[0])))
            .map(sample_pair_from_relative_preferences, tf.data.AUTOTUNE)
            .repeat(sample_freq)
            .shuffle(10_000)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    settings.init_ranker_input_path["train"]
    # get data generators
    train_ds = get_pairs_generator(settings.init_ranker_input_path["train"])
    val_ds = get_pairs_generator(settings.init_ranker_input_path["test"])
    # return train_ds
    
    ranker = RankNet()
    ranker.compile(optimizer='adam', loss='binary_crossentropy')

    ensure_dir_exists(settings.model_path)
    history = ranker.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=_EPOCHS,
        callbacks=[early_stopping, lr_reducer] # save_best_model
    )
    return ranker, history


if __name__ == "__main__":
    from mt.evaluation.simulation.data_utils import determine_features
    features = determine_features(*list(settings.libsvm_path.values()))
    _, history = fit_model(
        features,
        batch_size=_BATCH_SIZE,
        sample_freq=3
    )
    # plot loss history
    plot_metrics(history.history['loss'], history.history['val_loss'], "Loss", "Loss", ylim=1.0)