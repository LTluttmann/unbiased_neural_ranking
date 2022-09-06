import tensorflow as tf
import tensorflow_ranking as tfr


from mt.data.dataset.dataset_ops import BaseDatasetOp
from mt.config import config

from abc import ABC, abstractmethod
import warnings


class AbstractTfRecordParser(BaseDatasetOp, ABC):

    def __init__(self, feature_spec: dict, context_feature_spec: dict = None, on_batch=True) -> None:
        super().__init__(on_batch=on_batch)
        self.feature_spec = feature_spec.copy()
        self.context_feature_spec = context_feature_spec.copy() if context_feature_spec else {}

    def call_on_batch(self, x, *args, **kwargs):
        return self.parse_fn(x, *args, **kwargs)

    def call_on_single_example(self, x):
        return self.call_on_batch(x)

    @abstractmethod
    def parse_fn(self, raw_record, *args, **kwargs):
        pass

    @staticmethod
    def _get_scalar_default_value(dtype, default_value):
        """Gets the scalar compatible default value."""
        if dtype == tf.string:
            return default_value or ""
        elif default_value is None:
            return -1
        elif isinstance(default_value, int) or isinstance(default_value, float):
            return default_value
        elif (isinstance(default_value, list) or
                isinstance(default_value, tuple)) and len(default_value) == 1:
            return default_value[0]
        else:
            raise ValueError(
                "Only scalar or equivalent is allowed in default_value.")


class SeqExampleParser(AbstractTfRecordParser):

    def __init__(
            self,
            feature_spec: dict,
            context_feature_spec: dict = None,
            list_size: int = None, 
            on_batch=True) -> None:
        super().__init__(feature_spec, context_feature_spec, on_batch=on_batch)

        self.list_size = list_size
        self.check_feature_maps()

    def check_feature_maps(self):
        # batch mode -> calls tfranking parser
        if self.list_size:
            new_feature_map = {}
            for k, v in self.feature_spec.items():
                # determine default values to pad serps
                default_value = self._get_scalar_default_value(v.dtype, v.default_value)
                # somehow, the tfr parser needs the extra dimension, hence enforce it here! DO NOT REMOVE!
                new_feature_map[k] = tf.io.FixedLenFeature([1], dtype=v.dtype, default_value=default_value)
            self.feature_spec.update(new_feature_map)
        else:
            warning_message = """
                NOTE we cannot accept Ragged Tensors here because:
                1: tokenization on multi-dimensional Ragged-Tensors is not possible due to internal error
                2: sampling is not possible on ragged tensors.
                Hence, if batched operations should be used, pad the sequence using list_size argument.
            """
            new_feature_map = {}
            for k, v in self.feature_spec.items():
                if isinstance(v, tf.io.RaggedFeature):
                    warnings.warn(warning_message)
                if k in config.RAW_TEXT_COLUMNS:
                    new_feature_map[k] = tf.io.FixedLenSequenceFeature(
                        [], dtype=v.dtype, default_value=None)
                else:
                    new_feature_map[k] = tf.io.FixedLenSequenceFeature(
                        [], dtype=v.dtype, default_value=None)
            self.feature_spec.update(new_feature_map)

            new_context_map = {}
            for k, v in self.context_feature_spec.items():
                if isinstance(v, tf.io.RaggedFeature):
                    warnings.warn(warning_message)
                if k in config.RAW_TEXT_COLUMNS:
                    new_context_map[k] = tf.io.FixedLenFeature(
                        [], dtype=v.dtype, default_value=None)
                else:
                    new_context_map[k] = tf.io.FixedLenFeature(
                        [1], dtype=v.dtype, default_value=None)
            self.context_feature_spec.update(new_context_map)

    def parse_fn(self, raw_record: tf.Tensor) -> dict:
        """parses a single example from a tfrecord
        Args:
            raw_record (tf.Tensor): a zero dimensional Tensor, holding the encoded elwc
        Returns:
            Union[tf.Tensor, tf.Tensor]: two tensors, holding the features and the labels of the examples
                respectively.
        A typical input signature for batched input should look like so, i.e. used FixedLenFeatures for 
        the features with a default value
        context_map = {
            "searchterm": tf.io.FixedLenFeature([], tf.string)
        }
        feature_map = {
            "product_pd_Name": tf.io.FixedLenFeature([1], tf.string, default_value=""),
            "click": tf.io.FixedLenFeature([1], tf.int64, default_value=-1)
        }
        For unbatched data switch to FixedLenSequenceFeatures. No default value necessary
        context_map = {
            "searchterm": tf.io.FixedLenFeature([], tf.string)
        }
        feature_map = {
            "product_pd_Name": tf.io.FixedLenFeature([1], tf.string, default_value=""),
            "click": tf.io.FixedLenFeature([1], tf.int64, default_value=-1)
        }  
        """

        # if not raw_record.get_shape().as_list() and self.list_size:
        #     raise ValueError(
        #         "padding / truncation makes only sense when batches of data are provided.")
        if raw_record.get_shape().as_list() and not self.list_size:
            raise ValueError(
                "Batches of data may only be provided, if truncation/padding is enabled. Set list_size argument")

        if self.list_size:
            features = tfr.data.parse_from_sequence_example(
                raw_record, self.list_size, self.context_feature_spec, self.feature_spec)
            # function returns a empty dimension to the features. Squeeze it
            for k, t in features.items():
                features[k] = tf.squeeze(t, axis=-1)
        else:
            context, features, _ = tf.io.parse_sequence_example(
                raw_record, context_features=self.context_feature_spec, sequence_features=self.feature_spec
            )
            features.update(context)

        return features


class ExampleParser(AbstractTfRecordParser):

    def __init__(self, feature_spec: dict, on_batch=True) -> None:
        super().__init__(feature_spec, on_batch=on_batch)
        if on_batch:
            self.check_feature_maps_batched()
        else:
            self.check_feature_maps_unbatched()

    def check_feature_maps_unbatched(self):
        new_feature_map = {}
        for k, v in self.feature_spec.items():
            default_value = self._get_scalar_default_value(v.dtype, v.default_value)
            # we need that "[1]" to ensure second dimension (besides batch dim)
            new_feature_map[k] = tf.io.FixedLenFeature(
                [], dtype=v.dtype, default_value=default_value)
        self.feature_spec.update(new_feature_map)

    def check_feature_maps_batched(self):
        new_feature_map = {}
        for k, v in self.feature_spec.items():
            default_value = self._get_scalar_default_value(v.dtype, v.default_value)
            # we need that "[1]" to ensure second dimension (besides batch dim)
            new_feature_map[k] = tf.io.FixedLenFeature(
                [1], dtype=v.dtype, default_value=default_value)
        self.feature_spec.update(new_feature_map)

    def parse_fn(self, raw_record: tf.Tensor) -> dict:

        features = tf.io.parse_example(
            raw_record, features=self.feature_spec
        )

        return features
