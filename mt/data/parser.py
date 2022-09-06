import tensorflow as tf
import tensorflow_ranking as tfr

from mt.data.callbacks.callback import DatasetCallback
from mt.config import config

from abc import ABC, abstractmethod
from google.protobuf import descriptor_pb2 as pb
from typing import Union, List
import warnings


class AbstractBaseParser(ABC):

    def __init__(self, reader=tf.data.TFRecordDataset, callbacks:List[DatasetCallback] = None) -> None:
        super().__init__()
        self.reader = reader
        self.callbacks = callbacks if callbacks else []

    @abstractmethod
    def parse_fn(self, raw_record, *args, **kwargs):
        pass

    @tf.autograph.experimental.do_not_convert
    def parse_examples(self, dataset_w_raw, *args, **kwargs) -> tf.data.Dataset:
        # if isinstance(dataset_w_raw, BatchDataset):

        dataset = dataset_w_raw.map(lambda raw: self.parse_fn(
            raw, *args, **kwargs), num_parallel_calls=tf.data.AUTOTUNE)

        for callback in self.callbacks:
            dataset: tf.data.Dataset = dataset.map(callback.call_fn, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def load_and_parse_examples(self, filepattern, *args, **kwargs):
        dataset = tf.data.Dataset.list_files(
            filepattern).interleave(self.reader)
        dataset = self.parse_examples(dataset, *args, **kwargs)
        return dataset


class LibsvmParser(AbstractBaseParser):
    """Class for parsing an example list with context from a tfrecord
    """

    def __init__(self, features, max_rank, return_position_vector=True) -> None:
        super().__init__(reader=tf.data.TextLineDataset)
        self.features = features
        self.max_rank = max_rank
        self.return_position_vector = return_position_vector

    def parse_fn(self, line) -> Union[tf.Tensor, tf.Tensor]:
        # strip and split line
        line = tf.strings.strip(line)
        columns = tf.strings.split([line], ' ')
        # parse label
        labels = tf.strings.to_number(columns.values[0], out_type=tf.int32)
        labels = tf.reshape(labels, [-1])
        # parse qid
        qid = tf.strings.split(columns.values[1], ':')[-1]
        qid = tf.reshape(qid, [-1])
        # parse position
        pos = tf.strings.split(columns.values[2], ':')[-1]
        pos = tf.strings.to_number(pos, out_type=tf.int32)
        pos = tf.one_hot(pos, depth=self.max_rank)

        splits = tf.strings.split(columns.values[3:], ':')

        feat_ids, feat_vals = tf.split(
            splits.to_tensor(), num_or_size_splits=2, axis=1)
        feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int64)
        feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)

        sparse_feature = tf.SparseTensor(
            feat_ids, tf.reshape(feat_vals, [-1]), [max(self.features)])
        dense_feature = tf.sparse.to_dense(sparse_feature)

        if self.return_position_vector:
            return {"features": dense_feature, "position": pos}, labels
        else:
            return dense_feature, labels


class AbstractTfRecordParser(AbstractBaseParser):
    """tfrecord parser needs dictionaries mapping feature names to types"""

    def __init__(
            self,
            feature_spec: dict,
            context_feature_spec: dict = None,
            callbacks: List[DatasetCallback] = None) -> None:
        """class constructor

        Args:
            feature_spec (dict): A dictionary, mapping feature ids to tf feature types
            context_feature_spec (dict, optional): dictionary mapping keys to context features.
                Defaults to None, since context must not necessarily have features
        """
        super().__init__(reader=tf.data.TFRecordDataset, callbacks=callbacks)
        self.feature_spec = feature_spec
        self.context_feature_spec = context_feature_spec if context_feature_spec else {}
        # self.callbacks.append(self.SplitCallback(self))

    class SplitCallback(DatasetCallback):
        """splits context and examples"""
        def __init__(self, parser) -> None:
            super().__init__(batch_processing=True)
            self.parser = parser
        
        def _unbatch_operation(self, inputs):
            context = {k: v for k, v in inputs.items() if k in self.parser.context_feature_spec.keys() or k == config.QUERY_COL}
            example = {k: v for k, v in inputs.items() if k in self.parser.feature_spec.keys() and k != config.QUERY_COL}

            context["tokens"] = context.pop(config.QUERY_COL)
            example["tokens"] = example.pop(config.PRODUCT_TITLE_COL)

            return context, example   

        def _batch_operation(self, inputs):
            return self._unbatch_operation(inputs)


class ElwcParser(AbstractTfRecordParser):
    """Class for parsing an example list with context from a tfrecord
    """

    def __init__(
            self, 
            feature_spec: dict, 
            context_feature_spec: dict = None,
            callbacks: List[DatasetCallback] = None) -> None:
        """class for parsing data in example-list-with-context format
        """
        super().__init__(feature_spec, context_feature_spec, callbacks=callbacks)

    @staticmethod
    def _get_descriptor_set():
        """Returns a FileDescriptorSet proto to be used by tf.io.decode_proto."""
        proto = pb.FileDescriptorSet()

        # The FileDescriptor for tensorflow.ranking.internal.ExampleListWithContext.
        file_proto = proto.file.add(
            name="serialized_example_list.proto",
            package="tensorflow.ranking.internal",
            syntax="proto3")
        message_proto = file_proto.message_type.add(
            name="SerializedExampleListWithContext")
        message_proto.field.add(
            name="examples",
            number=1,
            type=pb.FieldDescriptorProto.TYPE_BYTES,
            label=pb.FieldDescriptorProto.LABEL_REPEATED
        )
        message_proto.field.add(
            name="context",
            number=2,
            type=pb.FieldDescriptorProto.TYPE_BYTES
        )

        return proto

    def _decode_as_serialized_example_list(self, serialized):
        """Decodes as `SerializedExampleListWithContext`."""
        serialized = tf.convert_to_tensor(value=serialized)
        sizes, (serialized_context, serialized_list) = tf.io.decode_proto(
            serialized,
            message_type="{}.{}".format(
                "tensorflow.ranking.internal", "SerializedExampleListWithContext"),
            field_names=["context", "examples"],
            output_types=[tf.string, tf.string],
            descriptor_source=(
                b"bytes://" + self._get_descriptor_set().SerializeToString())
        )
        # For batched inputs, sizes is of shape [batch_size, 2], for both context and
        # examples. We slice to obtain the size for example lists.
        if sizes.get_shape().rank == 2:
            sizes = sizes[:, 1]
        elif sizes.get_shape().rank == 1:  # parsing single ELWC.
            sizes = sizes[1]  # sizes is a tuple for context and examples.
        return serialized_context, serialized_list, sizes

    def parse_fn(self, raw_record: tf.Tensor, *args, **kwargs) -> dict:
        """parses a single example from a tfrecord

        Args:
            raw_record (tf.Tensor): a zero dimensional Tensor, holding the encoded elwc

        Returns:
            Union[tf.Tensor, tf.Tensor]: two tensors, holding the features and the labels of the examples
                respectively.
        """
        (serialized_context, serialized_list,
         _) = self._decode_as_serialized_example_list(raw_record)
        example_features = tf.compat.v1.io.parse_example(
            tf.reshape(serialized_list, [-1]), self.feature_spec
        )
        if self.context_feature_spec:
            example_features.update(
                tf.compat.v1.io.parse_example(
                    serialized_context,
                    self.context_feature_spec)
            )

        return example_features


class SeqExampleParser(AbstractTfRecordParser):

    def __init__(
            self,
            feature_spec: dict,
            context_feature_spec: dict = None,
            list_size: int = None,
            callbacks: List[DatasetCallback] = None) -> None:
        super().__init__(feature_spec, context_feature_spec, callbacks=callbacks)

        self.list_size = list_size
        self.check_feature_maps()

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

    def check_feature_maps(self):
        # batch mode -> calls tfranking parser
        if self.list_size:
            new_feature_map = {}
            for k, v in self.feature_spec.items():
                default_value = self._get_scalar_default_value(
                    v.dtype, v.default_value)
                new_feature_map[k] = tf.io.FixedLenFeature(
                    [1], dtype=v.dtype, default_value=default_value)
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

                new_feature_map[k] = tf.io.FixedLenSequenceFeature(
                    [], dtype=v.dtype, default_value=None)

            self.feature_spec.update(new_feature_map)

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

        if not raw_record.get_shape().as_list() and self.list_size:
            raise ValueError(
                "padding / truncation makes only sense when batches of data are provided.")
        elif raw_record.get_shape().as_list() and not self.list_size:
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

    def __init__(self, feature_spec: dict, callbacks: List[DatasetCallback] = None) -> None:
        super().__init__(feature_spec, callbacks=callbacks)
        self.check_feature_maps()

    def check_feature_maps(self):
        new_feature_map = {}
        for k, v in self.feature_spec.items():
            # we need that "[1]" to ensure second dimension (besides batch dim)
            new_feature_map[k] = tf.io.FixedLenFeature(
                [1], dtype=v.dtype, default_value=None)
        self.feature_spec.update(new_feature_map)

    def parse_fn(self, raw_record: tf.Tensor) -> dict:

        features = tf.io.parse_example(
            raw_record, features=self.feature_spec
        )

        return features
