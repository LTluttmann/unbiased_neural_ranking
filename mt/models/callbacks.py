import tensorflow as tf
import tensorflow.keras as nn
from tensorflow.keras.utils import Progbar

from mt.evaluation.embeddings.vis_embeddings import main as get_tsne_plot
from mt.utils import S3Url, ensure_url_format
from mt.models.model_io import copy_keras_model_to_s3, copy_keras_weights_to_s3
from mt.config import config

import pickle




class TransformerLRSchedule(nn.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, scale=1.0):
        super(TransformerLRSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.scale = scale
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return lr * self.scale


class PrintEpochCallback(nn.callbacks.Callback):

    def __init__(self, epochs, steps_train=None):
        super(PrintEpochCallback, self).__init__()
        self.epochs = epochs
    def on_epoch_begin(self, epoch, logs: dict=None):
        print("\nepoch {}/{}".format(epoch+1, self.epochs))


class SaveCallback(nn.callbacks.ModelCheckpoint):
    def __init__(self, local_filepath, model_attribute = None, model_name=None, s3_path:S3Url=None, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(local_filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)
        self.s3_path = ensure_url_format(s3_path)
        self.model_attribute = model_attribute
        self.model_name = model_name or config.ENCODER_FILENAME

    def set_model(self, model):
        if self.model_attribute:
            self.model = getattr(model, self.model_attribute)
        else:
            super().set_model(model)

    def on_train_end(self, logs=None):
        
        if self.s3_path:
            if self.save_weights_only:
                tf.print(f"copy weights to {self.s3_path.url}")
                copy_keras_weights_to_s3(self.filepath, self.s3_path.bucket, self.s3_path.key, model_name=self.model_name)                
            else:
                tf.print(f"copy model to {self.s3_path.url}")
                copy_keras_model_to_s3(self.filepath, self.s3_path.bucket, self.s3_path.key, model_name=self.model_name)


class SaveEncoderCallback(nn.callbacks.ModelCheckpoint):
    def __init__(self, local_filepath, model_name=None, s3_path:S3Url=None, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(local_filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)
        self.s3_path = ensure_url_format(s3_path)
        self.model_name = model_name or config.ENCODER_FILENAME

    def set_model(self, model):
        self.model = model.encoder

    def on_train_end(self, logs=None):
        
        if self.s3_path:
            tf.print(f"copy model to {self.s3_path.url}")
            copy_keras_model_to_s3(self.filepath, self.s3_path.bucket, self.s3_path.key, model_name=self.model_name)


class SaveRankerCallback(nn.callbacks.ModelCheckpoint):
    def __init__(self, local_filepath, model_name=None, s3_path:S3Url=None, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(local_filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)
        self.s3_path = ensure_url_format(s3_path)
        self.model_name = model_name or config.ENCODER_FILENAME

    def set_model(self, model):
        self.model = model.ranker

    def on_train_end(self, logs=None):
        
        if self.s3_path:
            tf.print(f"copy model to {self.s3_path.url}")
            copy_keras_model_to_s3(self.filepath, self.s3_path.bucket, self.s3_path.key, model_name=self.model_name)


class SaveModelCallback(nn.callbacks.ModelCheckpoint):
    def __init__(self, local_filepath, model_name=None, s3_path:S3Url=None, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super().__init__(local_filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)
        self.s3_path = ensure_url_format(s3_path)
        self.model_name = model_name or config.ENCODER_FILENAME

    def on_train_end(self, logs=None):
        
        if self.s3_path:
            tf.print(f"copy model to {self.s3_path.url}")
            copy_keras_model_to_s3(self.filepath, self.s3_path.bucket, self.s3_path.key, model_name=self.model_name)

class ErrorLogsCallback(nn.callbacks.Callback):

    def __init__(self, epochs, steps_train=None):
        super(ErrorLogsCallback, self).__init__()
        self.epochs = epochs
        self.steps_train = steps_train

    def _update_losses(self, logs, batch):
        for key, loss in logs.items():
            mean_loss = self.mean_losses.get(key, None)
            self.mean_losses[key] = loss if not mean_loss else (batch * mean_loss + loss) / (batch+1)
            logs[key] = self.mean_losses[key]

    def on_epoch_begin(self, epoch, logs: dict=None):
        print("\nepoch {}/{}".format(epoch+1, self.epochs))
        self.pb = Progbar(self.steps_train)
        self.mean_losses = {}

    def on_train_batch_end(self, batch, logs: dict=None):
        self.steps_train = max(self.steps_train or 0, (batch+1))
        self._update_losses(logs, batch)
        self.pb.update(batch, values=[(key, self.mean_losses[key]) for key in logs.keys()])

    def on_test_batch_end(self, batch, logs=None):
        self._update_losses(logs, batch)

    def on_epoch_end(self, epoch, logs=None):
        self.pb.update(self.steps_train+1, values=[(key, self.mean_losses[key]) for key in logs.keys()])


class LogHistoryCallback(nn.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        
        with open(f'hist_{self.model.name}_after_{epoch}', 'wb') as file_pi:
            pickle.dump(self.model.history.history, file_pi)


class VisualizeEmbeddingsCallback(nn.callbacks.Callback):
    def __init__(self, tokenizer_callback, num_products:int=30):
        super().__init__()
        self.tokenizer = tokenizer_callback
        self.num_products = num_products

    def on_train_end(self, logs=None):
        get_tsne_plot(self.model.encoder, self.tokenizer, self.num_products)


class LossHistory(nn.callbacks.Callback):
    def __init__(self, record_every_n: int = 100, val_ds=None):
        super().__init__()
        self.record_every_n = record_every_n
        self.val_ds = val_ds

    def _calc_validation_loss(self):
        losses = []
        for inputs in iter(self.val_ds):
            metrics = self.model._validate_on_batch(inputs)
            losses.append(metrics.get("loss"))
        losses = tf.concat(losses, axis=0)
        return tf.reduce_mean(losses)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.history = {}
        self.cnt = 0
        self.rolling_mean_loss = 0

    def on_batch_end(self, batch, logs={}):
        self.cnt += 1
        loss = logs.get('loss')
        self.rolling_mean_loss = (self.rolling_mean_loss * (self.cnt-1) + loss) / self.cnt
        if self.cnt % self.record_every_n == 0:
            self.losses.append(self.rolling_mean_loss)
            if self.val_ds is not None:
                self.val_losses.append(self._calc_validation_loss())
            else:
                self.val_losses.append(None)
        

    def on_train_end(self, logs=None):
        history = {
            "loss": self.losses,
            "val_loss": self.val_losses}

        with open(f'{self.model.name}_history_after_{self.cnt}', 'wb') as file_pi:
            pickle.dump(history, file_pi)
        