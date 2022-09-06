import tensorflow as tf

from mt.data.dataset.dataset_ops import BaseDatasetOp
from mt.config import config

import itertools
from typing import List, Dict, Callable


class FeatureTransformationCallback(BaseDatasetOp):
    def __init__(self, column_operation_mappings: List[Dict[str, Callable]], on_batch=True) -> None:
        super().__init__(on_batch=on_batch)
        self.column_operation_mappings = [column_operation_mappings] if not isinstance(
            column_operation_mappings, list) else column_operation_mappings

    def call_on_single_example(self, inputs):
        x = inputs.copy()
        for column_operation_mapping in self.column_operation_mappings:
            for col, transformation_fn in column_operation_mapping.items():
                x[col] = transformation_fn(x[col])
        return x

    def call_on_batch(self, inputs):
        return self.call_on_single_example(inputs)


class FeatureMerger(BaseDatasetOp):
    def __init__(self, merge_cols, merged_feature_name, on_batch = True) -> None:
        super().__init__(on_batch=on_batch)
        self.merge_cols = merge_cols
        print("merging the following columns to a single feature tensor: ", "\n".join(self.merge_cols))
        self.merged_feature_name = merged_feature_name

    def call_on_single_example(self, inputs):
        x = inputs.copy()
        merge_tensors = []
        for col in self.merge_cols:
            feature = tf.cast(x.pop(col), tf.float32)
            feature = tf.reshape(feature, (-1, ))
            merge_tensors.append(feature)
        x[self.merged_feature_name] = tf.concat(merge_tensors, axis=-1)
        return x

    def call_on_batch(self, inputs):
        x = inputs.copy()
        merge_tensors = []
        for col in self.merge_cols:
            feature = tf.cast(x.pop(col), tf.float32)
            merge_tensors.append(feature)
        x[self.merged_feature_name] = tf.concat(merge_tensors, axis=-1)
        return x


class LayoutEncoder(BaseDatasetOp):
    
    devices = ["mobile", "desktop", "tablet"]
    layout_types = ["galleryPortrait", "list", "galleryLandscape", "galleryQuad"]
    positions = [str(x) for x in range(0, config.MAX_SEQ_LENGTH)]
    
    column_map = {
        "position": positions,
        "device": devices, 
        "layout_type": layout_types
    }
    
    _SEPERATOR = "+"
    _DEFAULT_OPS: Callable = staticmethod(lambda x: tf.expand_dims(x, -1))
    
    def __init__(
            self, 
            interaction_cols:list, 
            merge_cols:list=config.POSITION_BIAS_FEATURES, 
            feature_name:str="pb_features", 
            interaction_hashing:str="embedding", 
            column_operation_mapping: Dict[str, Callable] = {},
            embedding_dim=50, 
            on_batch = True) -> None:
        """
        mode: one of "embedding" or "one_hot"
        """
        super().__init__(on_batch=on_batch)
        
        self.interaction_cols = interaction_cols if interaction_cols else []
        self.merge_cols = [col for col in merge_cols if col not in self.interaction_cols]
        self.feature_name = feature_name
        self.lookup = tf.keras.layers.StringLookup(vocabulary=self.layouts)
        
        self.mapping = lambda x: tf.one_hot(x-1, len(self.layouts))  if interaction_hashing=="one_hot"  \
            else tf.keras.layers.Embedding(len(self.layouts)+1, embedding_dim)(x)
        # + 1 for unknown (always needed in embedding layers since string lookup starts counting at 1
        self.column_operation_mapping = column_operation_mapping
        
    @property
    def layouts(self):
        lists = [self.column_map[col] for col in self.interaction_cols]
        layout_list = []
        for combination in itertools.product(*lists):
            layout_list.append(self._SEPERATOR.join(combination))
        return layout_list

    def call_on_single_example(self, inputs):
        return self.call_on_batch(inputs)

    def call_on_batch(self, inputs):
        x = inputs.copy()
        merge_tensors = []
        if self.interaction_cols:
            interaction_tensors = []
            for col in self.interaction_cols:
                feature = x.pop(col)
                feature = tf.expand_dims(feature, -1)
                if feature.dtype != tf.string:
                    feature = tf.maximum(feature, 0)
                    feature = tf.strings.as_string(tf.cast(feature, tf.int64))
                interaction_tensors.append(feature)
            layout_combination = tf.strings.reduce_join(tf.concat(interaction_tensors, axis=-1), separator=self._SEPERATOR, keepdims=False, axis=-1)

            layout_hashes = self.lookup(layout_combination)
            layout_embeddings = self.mapping(layout_hashes)
        
            merge_tensors.append(layout_embeddings)
            
        for col in self.merge_cols:
            feature = x.pop(col)
            feature = self.column_operation_mapping.get(col, self._DEFAULT_OPS)(feature)
            merge_tensors.append(tf.cast(feature, tf.float32))      
           

        x[self.feature_name] = tf.concat(merge_tensors, axis=-1)
        return x


class NormalizeAlongAxis(BaseDatasetOp):

    def __init__(self, column, axis, mask_value=None, kind="z_norm", on_batch: bool = True) -> None:
        super().__init__(on_batch)
        self.column = column
        self.axis = axis
        self.mask_value = mask_value
        self.op = self._mapper(kind)

    def _mapper(self, kind):
        mapping = {
            "z_norm": self._z_norm,
            "min_max": self._min_max
        }
        op = mapping.get(kind, None)
        if op is None:
            raise ValueError(f"provide either z_norm or min_max as normalization function. Got {kind}")
        return op

    def _z_norm(self, vals, mask=None):
        if mask is not None:
            masked_vals = tf.ragged.boolean_mask(vals, mask)
            mean = tf.reduce_mean(masked_vals, keepdims=True, axis=self.axis).to_tensor()
            std = tf.sqrt(tf.reduce_mean(tf.ragged.boolean_mask(tf.square(
                vals-mean), mask), axis=self.axis, keepdims=True)+tf.keras.backend.epsilon()).to_tensor()
        else:
            mean = tf.reduce_mean(vals, keepdims=True, axis=self.axis)
            std = tf.sqrt(tf.reduce_mean(tf.square(vals-mean), axis=self.axis, keepdims=True)+tf.keras.backend.epsilon())
        return (vals - mean)/std

    def _min_max(self, vals, mask=None):
        if mask is not None:
            masked_vals = tf.ragged.boolean_mask(vals, mask)
            min_ = tf.reduce_min(masked_vals, keepdims=True, axis=self.axis)
            max_ = tf.reduce_max(masked_vals, keepdims=True, axis=self.axis)
        else:
            min_ = tf.reduce_min(vals, keepdims=True, axis=self.axis)
            max_ = tf.reduce_max(vals, keepdims=True, axis=self.axis)
        normed_vals = tf.where(tf.equal(max_, min_), vals, (vals - min_)/ (max_ - min_))
        return normed_vals

    def call_on_single_example(self, x):
        return self.call_on_batch(x)

    def call_on_batch(self, x):
        vals = x[self.column]
        if self.mask_value:
            mask = tf.not_equal(vals, self.mask_value)
            normed_vals = self.op(vals, mask)
            x[self.column] = tf.where(mask, normed_vals, vals)
        else:
            x[self.column] = self.op(vals)
        return x