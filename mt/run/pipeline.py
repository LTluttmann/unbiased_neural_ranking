from mt.data.dataset.callbacks.tokenizer_callback import TokenizerCallback
from mt.data.dataset.callbacks.feature_callbacks import FeatureMerger, NormalizeAlongAxis, LayoutEncoder, FeatureTransformationCallback
from mt.data.dataset.parser import SeqExampleParser, ExampleParser
from mt.data.dataset.sampler import SequenceExampleSampler
from mt.data.dataset.utils import concat_from_zipped_datasets, SampleWeighter, has_positives_and_negatives, calc_label_distribution
from mt.data.utils import get_blocklist

from mt.models.lse.encoder import DSSM, AttnDSSM, USE
from mt.models.ltr import espec, espec_val, cspec
from mt.models.ltr.attnrank import AttnRank
from mt.models.ltr.mlp import MLPRank
from mt.models.ltr.drmm import DHRMM
from mt.models.ltr.knrm import HybridKNRM
from mt.models.model_io import s3_get_keras_model
from mt.models.callbacks import TransformerLRSchedule, SaveCallback

from mt.models.ultr.joe import JointEstimator
from mt.tokenizer.tokenizer_io import load_bert_tokenizer_from_vocab_path, get_vocab_from_s3
from mt.utils import ensure_url_format
from mt.config import config

import tensorflow as tf
import tensorflow.keras as nn

from abc import ABC, abstractmethod
from typing import Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
from mt.evaluation.utils.inspection_utils import prepare_tokens_for_print
import warnings


@dataclass
class PipelineConfig:
    num_negatives: int
    random_negatives: int
    list_size: int
    num_tasks: int
    epochs: int
    batch_size: int
    learning_rate: Union[float, nn.optimizers.schedules.LearningRateSchedule]
    sampling_weight_col: str = None
    sample_w_replacement: bool = False
    normalize_input: bool = False
    val_metric_to_monitor: str = "val_loss"
    loss_weights: dict=None

    @property
    def mode(self):
        if self.val_metric_to_monitor == "val_loss":
            return "min"
        else:
            return "max"


class BasePipeline(ABC):

    def __init__(self,
                 model_url,
                 vocab_url, 
                 pbk_url,
                 pipeline_config: PipelineConfig,
                 ) -> None:

        self.pipeline_config = pipeline_config

        self.tokenizer, self.vocab = load_bert_tokenizer_from_vocab_path(vocab_url, return_vocab=True)
        self.pbks = get_vocab_from_s3(pbk_url)
        self.pbk_lookup = tf.keras.layers.StringLookup(vocabulary=self.pbks)

        self.model_url = ensure_url_format(model_url)

        self.tokenizer_callback = self.get_tokenizer()


    def get_tokenizer(self):
   
        if not config.MAX_TOKENS:
            seq_length=None
        else:
            seq_length={config.QUERY_COL: config.MAX_TOKENS, 
                        config.PRODUCT_TITLE_COL: config.MAX_TOKENS}

        tokenizer_callback = TokenizerCallback(tokenizer=self.tokenizer, 
                                      cols=[config.QUERY_COL, config.PRODUCT_TITLE_COL],
                                      max_length=seq_length)

        return tokenizer_callback

    @abstractmethod 
    def training_dataset(self, train_path):
        pass

    @abstractmethod 
    def validation_dataset(self, val_path):
        pass

    @abstractmethod
    def get_model(self, name, *args, **kwargs):
        pass

    @abstractmethod
    def start(self, callbacks, **kwargs):
        pass


class BaseEncoderPipeline(BasePipeline):

    def __init__(self, 
                 model_url, 
                 encoder_name,
                 vocab_url, 
                 pbk_url, 
                 pipeline_config: PipelineConfig, 
                 embedder_name=None) -> None:

        super().__init__(model_url, vocab_url, pbk_url, pipeline_config, embedder_name)

        self.encoder, self.optimizer = self.get_encoder(encoder_name)

    def get_model(self, name, learning_rate):

        def get_use():
            d_model = 256
            encoder = USE(len(self.vocab), embedding_dim=d_model, dff=1024, num_attn_heads=4,
                        num_attn_layers=2, dropout_rate=0.1)
            learning_rate = TransformerLRSchedule(d_model)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            return encoder, optimizer

        def get_dssm():
            encoder = DSSM(len(self.vocab), embedding_dim=300, dense_layer_nodes=[300, 300],
                           batch_norm=True, dropout_rate=0.3)

            optimizer = nn.optimizers.Adam(learning_rate=learning_rate, clipnorm=3.0)
            return encoder, optimizer

        def get_dssm_w_attn():
            encoder = AttnDSSM(len(self.vocab), embedding_dim=300, sentence_emb_dim=128)
            optimizer = nn.optimizers.Adam(learning_rate=learning_rate, clipnorm=3.0)
            return encoder, optimizer

        if name == "dssm":
            encoder, optimizer = get_dssm()
        elif name == "use":
            encoder, optimizer = get_use()
        elif name == "dssm_w_attn":
            encoder, optimizer = get_dssm_w_attn()

        return encoder, optimizer


class BaseLTRPipeline(BasePipeline):

    def __init__(self, 
                 model_url, 
                 vocab_url, 
                 pbk_url, 
                 pipeline_config: PipelineConfig, 
                 encoder_name=None, 
                 classifier_name=None, 
                 embedder_name=None) -> None:

        super().__init__(model_url, vocab_url, pbk_url, pipeline_config)

        if encoder_name:
            self.encoder = s3_get_keras_model(encoder_name, self.model_url)

        if classifier_name:
            self.pbk_classifier = s3_get_keras_model(classifier_name, self.model_url)

        if embedder_name:
            embedder = s3_get_keras_model(embedder_name, self.model_url)
            self.embedding_layer = embedder.embedding_layer
        else:
            self.embedding_layer = None

        self.feature_merger = FeatureMerger(config.MERGE_COLS, merged_feature_name=config.NUMERIC_FEATURES_COL)

        self.sequence_parser = SeqExampleParser(espec, cspec, list_size=self.pipeline_config.list_size)

        if self.pipeline_config.normalize_input:
            warnings.warn("input normalization is activated")
            self.normalize_num_features = NormalizeAlongAxis(column=config.NUMERIC_FEATURES_COL, axis=[0,1], mask_value=-1, kind="z_norm")
        elif self.pipeline_config.num_negatives == -1:
            warnings.warn("""input is not normalized, however no sampling is performed. 
            This can lead to numerical issues, since BatchNormalization will not take 
            into account the masking!""")

        # NOTE 72 should be enough for validation set
        self.sequence_parser_val = SeqExampleParser(espec_val, cspec, list_size=config.MAX_SEQ_LENGTH)

        self.example_parser = ExampleParser(espec)

        self.sampler = SequenceExampleSampler(num_negatives=self.pipeline_config.num_negatives, 
                                              replacement=self.pipeline_config.sample_w_replacement,
                                              sample_weight=self.pipeline_config.sampling_weight_col)

        self.feature_transformer = self.get_feature_transformer()
        self.layout_encoder = self.get_layout_encoder()

        self.sample_weighter = SampleWeighter(self.pipeline_config.num_tasks, self.pipeline_config.loss_weights)


    def get_feature_transformer(self):

        feature_transformations1 = {
            k: lambda x: tf.where(tf.not_equal(x, -1),
                                  tf.math.log1p(tf.maximum(0.0, tf.cast(x, tf.float32))), 
                                  tf.cast(x, tf.float32))
            for k in config.LOG1P_TRANSFORM_COLS
        }

        feature_transformations2 = {k: lambda x: tf.expand_dims(x, -1) for k in config.NUMERICAL_COLUMNS}

        one_hot_transformation = []
        for k,v in config.CATEGORICAL_FEATURES.items():
            lookup = tf.keras.layers.StringLookup(vocabulary=v)
            transformation = {k: lambda x: tf.one_hot(lookup(x)-1, len(v))}
            one_hot_transformation.append(transformation)

        transform_callback = FeatureTransformationCallback(column_operation_mappings=[feature_transformations1,
                                                                                      feature_transformations2,
                                                                                      *one_hot_transformation])
        return transform_callback


    def get_layout_encoder(self):
        device_lookup = tf.keras.layers.StringLookup(vocabulary=config.DEVICE_VOCAB)
        layout_lookup = tf.keras.layers.StringLookup(vocabulary=config.LAYOUT_VOCAB)

        layout_kwargs = {"interaction_cols": None,
                         "merge_cols": config.POSITION_BIAS_FEATURES,
                         "column_operation_mapping": {
                             # position starts with one, hence subtract by one, since one_hot_expects values starting from 0
                             config.POSITION_COL: lambda x: tf.one_hot(tf.cast(x-1, tf.int64), config.MAX_SEQ_LENGTH),
                             config.DEVICE_COL: lambda x: tf.one_hot(device_lookup(x)-1, len(config.DEVICE_VOCAB)),
                             config.LAYOUT_COL: lambda x: tf.one_hot(
                                 layout_lookup(x)-1, len(config.LAYOUT_VOCAB))
                         },
                         "feature_name": config.POS_BIAS_FEATURE_COL}

        layout_encoder = LayoutEncoder(**layout_kwargs)

        return layout_encoder

    def get_model(self, name, **kwargs):

        def get_attnrank():
            attnrank = AttnRank(self.encoder,
                            classifier=self.pbk_classifier,
                            pbks=self.pbks,
                            embedding_layer=self.embedding_layer,
                            num_tasks=self.pipeline_config.num_tasks,
                            **kwargs) # .build_graph() # NOTE: build_graph is important as it ensures dynamic shape after loading
            return attnrank

        def get_mlp():
            mlp = MLPRank(self.encoder,
                          classifier=self.pbk_classifier,
                          pbks=self.pbks,
                          num_tasks=self.pipeline_config.num_tasks,
                          **kwargs) # NOTE: build_graph should be implemented
            return mlp

        def get_drmm():
            drmm = DHRMM(self.encoder.embedding_layer,
                         pbk_classifier=self.pbk_classifier, 
                         pbks=self.pbks, 
                         **kwargs) # NOTE: build_graph should be implemented
            return drmm

        def get_knrm():
            drmm = HybridKNRM(self.encoder.embedding_layer,
                            pbk_classifier=self.pbk_classifier, 
                            pbks=self.pbks, 
                            **kwargs) # NOTE: build_graph should be implemented
            return drmm

        if name == "attnrank":
            ranker = get_attnrank()
        elif name == "mlp":
            ranker = get_mlp()
        elif name == "drmm":
            ranker = get_drmm()        
        elif name == "knrm":
            ranker = get_knrm()
        return ranker

    @abstractmethod 
    def training_dataset(self, *args, **kwargs):
        pass

    @abstractmethod 
    def validation_dataset(self, *args, **kwargs):
        pass


class LTRPipeline(BaseLTRPipeline):
    def __init__(self,
                 model_url, 
                 vocab_url, 
                 pbk_url, 
                 pipeline_config: PipelineConfig, 
                 encoder_name=None, 
                 classifier_name=None, 
                 embedder_name=None) -> None:
        super().__init__(model_url=model_url,
                         encoder_name=encoder_name,
                         classifier_name=classifier_name,
                         vocab_url=vocab_url, pbk_url=pbk_url,
                         pipeline_config=pipeline_config,
                         embedder_name=embedder_name)


    def training_dataset(self, *args, **kwargs):
        if self.pipeline_config.random_negatives > 0:
            return self._training_dataset_w_additional_negatives(*args, **kwargs)
        else:
            return self._training_dataset(*args, **kwargs)

    def validation_dataset(self):
        
        normalize_judgements = NormalizeAlongAxis(column=config.JUDGEMENT_COL,
                                                  axis=1, mask_value=-1, kind="min_max")

        ds_val = (tf.data.Dataset.list_files("training_data/val/*.tfrecord")
                  .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
                  .shuffle(10_000)
                  .batch(50)
                  .map(self.sequence_parser_val, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.tokenizer_callback, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.feature_transformer, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.feature_merger, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.layout_encoder, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(normalize_judgements, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(calc_label_distribution, num_parallel_calls=tf.data.AUTOTUNE))

        if hasattr(self, "normalize_num_features"):
            # add the normalization callback if it exists
            ds_val = ds_val.map(self.normalize_num_features,
                                num_parallel_calls=tf.data.AUTOTUNE)

        return ds_val


    def _training_dataset(self):
        
        ds = (tf.data.Dataset.list_files("training_data/train/*.tfrecord", shuffle=True)
              .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(100_000)
              .batch(self.pipeline_config.batch_size)
              .map(self.sequence_parser, num_parallel_calls=tf.data.AUTOTUNE))

        if self.pipeline_config.num_negatives != -1:
            # if we sample a positive along with several negatives, we need to make sure 
            # that there are actually positive and negative examples in the sequence
            ds = (ds.unbatch()
                  .filter(has_positives_and_negatives)
                  .batch(self.pipeline_config.batch_size))

        ds = (ds.map(self.sampler, num_parallel_calls=tf.data.AUTOTUNE)
              .map(self.tokenizer_callback, num_parallel_calls=tf.data.AUTOTUNE)
              .map(self.sample_weighter, num_parallel_calls=tf.data.AUTOTUNE)
              .map(self.feature_transformer, num_parallel_calls=tf.data.AUTOTUNE)
              .map(self.feature_merger, num_parallel_calls=tf.data.AUTOTUNE)
              .map(self.layout_encoder, num_parallel_calls=tf.data.AUTOTUNE)
              .map(calc_label_distribution, num_parallel_calls=tf.data.AUTOTUNE))

        if hasattr(self, "normalize_num_features"):
            # add the normalization callback if it exists
            ds = ds.map(self.normalize_num_features,
                        num_parallel_calls=tf.data.AUTOTUNE)

        return ds.prefetch(tf.data.AUTOTUNE)

    def _training_dataset_w_additional_negatives(self):
        # sample random negatives
        ds_negs = (tf.data.Dataset.list_files("click/train/*.tfrecord", shuffle=True)
                   .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
                   .shuffle(1_000_000)
                   .map(self.example_parser, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(self.pipeline_config.random_negatives))

        ds = (tf.data.Dataset.list_files("training_data/train/*.tfrecord", shuffle=True)
              .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(100_000)
              .batch(self.pipeline_config.batch_size)
              .map(self.sequence_parser, num_parallel_calls=tf.data.AUTOTUNE))

        if self.pipeline_config.num_negatives != -1:
            # if we sample a positive along with several negatives, we need to make sure 
            # that there are actually positive and negative examples in the sequence
            ds = (ds.unbatch()
                  .filter(has_positives_and_negatives)
                  .batch(self.pipeline_config.batch_size))

        ds_zip = (tf.data.Dataset.zip((ds, ds_negs))
                  .map(concat_from_zipped_datasets, num_parallel_calls=tf.data.AUTOTUNE)
                  #.map(string_to_number, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.sampler, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.tokenizer_callback, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.sample_weighter, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.feature_transformer, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.feature_merger, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(self.layout_encoder, num_parallel_calls=tf.data.AUTOTUNE)
                  .map(calc_label_distribution, num_parallel_calls=tf.data.AUTOTUNE))

        if hasattr(self, "normalize_num_features"):
            # add the normalization callback if it exists
            ds_zip = ds_zip.map(self.normalize_num_features,
                                num_parallel_calls=tf.data.AUTOTUNE)

        return ds_zip.prefetch(tf.data.AUTOTUNE)


    def start(self, estimator, callbacks:list=None, train_steps=None, validation_steps=None):

        save_ranker = SaveCallback(estimator.ranker.name,
                                   model_attribute="ranker",
                                   model_name=estimator.ranker.name,
                                   s3_path=self.model_url,
                                   verbose=1,
                                   save_best_only=True,
                                   monitor=self.pipeline_config.val_metric_to_monitor,
                                   mode=self.pipeline_config.mode)

        save_bias_model = SaveCallback(estimator.propensity_estimator.name,
                                   model_attribute="propensity_estimator",
                                   model_name=estimator.propensity_estimator.name,
                                   s3_path=self.model_url,
                                   verbose=0,
                                   save_best_only=True,
                                   monitor=self.pipeline_config.val_metric_to_monitor,
                                   mode=self.pipeline_config.mode)
                                         

        if not callbacks:
            callbacks = [save_ranker, save_bias_model]
        else:
            callbacks.append(save_ranker)
            callbacks.append(save_bias_model)

        train_ds = self.training_dataset()
        if train_steps:
            train_ds = train_ds.take(train_steps)
        val_ds = self.validation_dataset()
        if validation_steps:
            val_ds = val_ds.take(validation_steps).cache()

        return estimator.fit(train_ds, validation_data=val_ds, epochs=self.pipeline_config.epochs, callbacks=callbacks)


    def top5k_dataset(self):
        eval_espec = espec.copy()
        eval_espec = {k:v for k,v in eval_espec.items() if k in config.RANKER_FEATURES + [config.OFFER_OR_PRODUCT_COL]}
        eval_cspec = cspec.copy()
        eval_cspec["searchterm_normalized"] = eval_cspec.get("searchterm")
        eval_parser = SeqExampleParser(eval_espec, eval_cspec)

        ds_top5k = (tf.data.Dataset.list_files("top5k/*.tfrecord")
                    .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
                    .map(eval_parser, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(1)
                    #.map(string_to_number, num_parallel_calls=tf.data.AUTOTUNE)
                    .map(self.tokenizer_callback, num_parallel_calls=tf.data.AUTOTUNE)
                    .map(self.feature_transformer, num_parallel_calls=tf.data.AUTOTUNE)
                    .map(self.feature_merger, num_parallel_calls=tf.data.AUTOTUNE))

        if hasattr(self, "normalize_num_features"):
            ds_top5k = ds_top5k.map(self.normalize_num_features, num_parallel_calls=tf.data.AUTOTUNE)

        return ds_top5k

    def get_5k_rankings(self, ranker, weights=None):

        ds_top5k = self.top5k_dataset()

        block_list = get_blocklist(config.BLOCKLIST_PATH)
        queries_with_distorting_retrievals = ["pc set komplett", "pc set komplett com", "komplett pc set"]

        dfs = []
        for x in tqdm(iter(ds_top5k)):

            query_normalized = x[config.NORMALIZED_QUERY_COL][0].numpy().decode("utf-8")
            
            if query_normalized in block_list:
                continue
            
            query = prepare_tokens_for_print(x[config.QUERY_COL], self.tokenizer)[0]
            
            if query in queries_with_distorting_retrievals:
                continue

            if hasattr(ranker, "input_signature"):
                inputs = {k: v for k,v in x.items() if k in ranker.input_signature.keys()}
            else:
                inputs = x 

            pred = ranker.predict(inputs, verbose=0)

            if isinstance(pred, list):
                if not weights:
                    weights = [1] * len(pred)
                pred = tf.add_n([weights[i] * pred[i] for i in range(len(pred))])

            pred = pred[0]
            docs = prepare_tokens_for_print(x[config.PRODUCT_TITLE_COL][0], self.tokenizer)
            pids = x[config.OFFER_OR_PRODUCT_COL][0]
            pids = [p.decode("utf-8") for p in pids.numpy()]

            df = pd.DataFrame({
                "query": query_normalized,
                "offer_or_product_id": pids,
                "title": docs,
                "score": pred
            })

            dfs.append(df)
            
        merged_df = pd.concat(dfs, ignore_index=True)

        df_agg = merged_df.groupby(["query", "offer_or_product_id"]).agg({"score": np.max, "title":lambda x: x.iloc[0]})
        df_sorted = df_agg.sort_values(["query", "score"], ascending=False)
        final_df = df_sorted.reset_index()
        if weights is not None:
            final_df.to_csv(f"5k_{ranker.name}_{'_'.join([str(x) for x in weights])}_weights.csv")
        else:
            final_df.to_csv(f"5k_{ranker.name}.csv")

        return merged_df
