import tensorflow as tf
import tensorflow.keras as nn

from mt.models.losses import binary_crossentropy_loss, triplet_loss, bpr_max, softmax_crossentropy_loss_new, sigmoid_crossentropy_loss
from mt.config import config

import abc
from typing import Callable, List
from collections import namedtuple

from tensorflow.keras.initializers import Constant, RandomUniform
from tensorflow.keras import backend as K


# Custom loss layer
class LearnedMultiLossLayer(nn.layers.Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(LearnedMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(LearnedMultiLossLayer, self).build(input_shape)

    def call(self, losses, training=True):
        assert len(losses) == self.nb_outputs
        total_loss = 0
        for loss, log_var in zip(losses, self.log_vars):
            precision = K.exp(-log_var[0])
            total_loss += precision * loss + log_var[0]
        return total_loss

class BalancedMultiLossLayer(nn.layers.Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = 2
        self.is_placeholder = True
        super(BalancedMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars

        self.lambda_weight = [self.add_weight(name='lambda_weight', shape=(),
                                            initializer=Constant(0.5), trainable=True)]

        super(BalancedMultiLossLayer, self).build(input_shape)

    def call(self, losses, training=True):
        assert len(losses) == 2
        total_loss = 0

        L1 = losses[0] * self.lambda_weight[0]
        L2 = losses[1] * (1-self.lambda_weight[0])
        if training:
            regularization = K.abs(L1-L2)
            total_loss = L1 + L2 + regularization
        else:
            total_loss = L1 + L2

        return total_loss


class RandomLossWeighter(nn.layers.Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        super(RandomLossWeighter, self).__init__(**kwargs)

    def call(self, losses, training=True):
        if training:
            weights = tf.nn.softmax(tf.random.normal((self.nb_outputs,)))
        else:
            weights = tf.ones(shape=(self.nb_outputs, ), dtype=tf.float32)
        total_loss = tf.add_n([weights[i] * losses[i] for i in range(self.nb_outputs)])
        return total_loss


Metric = namedtuple("MetricWrapper", "label pred metric_fn")


class BaseTrainer(abc.ABC, nn.Model):
    def __init__(
        self, 
        ranker: nn.Model, 
        losses:dict = None,
        learning_rate:float = 1e-3,
        sample_weight_col: str = None,
        num_negatives:int = None, 
        graph_exec:bool = True, 
        train_end_to_end:bool = True,
        loss_weights:list=None,
        loss_weighting_strategy: str="fixed",
        metrics:List[Metric] = None,
        clipnorm: float = 2.0,
        **kwargs):
        """Trains a given ranker

        Args:
            ranker (nn.Model): The ranking model to be trained

            num_negatives (int, optional): Specifies, how many negative samples should be considered 
            during training. 
            - For pointwise loss, set this to None. In that case, a label has to be passed to the trainer
            and the algorithm esentially performs a CTR prediction
            - For pairwise training with hinge loss provide num_negatives=1. The positive instance is 
            projected closer to the query than the negative instance
            - For pairwise training with softmax cross-entropy loss provide num_negatives > 1. The positive
            instance has to be at the first position 
            - For listwise training with softmax cross-entrioy loss (like above, but taking into account 
            the whole list), pass num_negatives = -1. A label has to be provided, since a SERP can contain
            several positives (clicks).
            Defaults to None.

            graph_exec (bool, optional): Whether to execute the training step in graph mode. Defaults to True.

            train_end_to_end (bool, optional): Whether or not to freeze the encoder during training. Defaults to True (not freeze).
        """
        super().__init__()
        self.ranker = ranker

        if hasattr(self.ranker, "encoder") and not train_end_to_end:
            self.ranker.encoder.trainable = False

        self.num_negatives = num_negatives
        self.graph_exec = graph_exec

        if losses is None:
            self.loss_fn = self._infer_loss_fn(**kwargs)

        self.eval_metrics = metrics

        self.compile(optimizer=nn.optimizers.Adam(learning_rate, clipnorm=clipnorm), loss=losses)

        if isinstance(self.loss, dict):
            if loss_weighting_strategy == "random":
                self.loss_weighter = RandomLossWeighter(len(self.loss))
            elif loss_weighting_strategy == "learned":
                self.loss_weighter = LearnedMultiLossLayer(len(self.loss))
            elif loss_weighting_strategy == "balanced":
                self.loss_weighter = BalancedMultiLossLayer(len(self.loss))
            elif loss_weighting_strategy == "fixed":
                loss_weights= [1.0] * len(self.loss) if loss_weights is None else loss_weights
                self.loss_weighter = lambda loss, _: sum(loss*loss_weights[i] for i, loss in enumerate(loss))
            else:
                raise ValueError(f"provide either random, balanced, learned or fixed as loss weighting strategy, got {loss_weighting_strategy}")

        self.sample_weight_col = sample_weight_col

    def _update_loss_from_dict(self, inputs, outputs, training=True):
        losses = []
        # total_loss = 0
        for key, loss_fn in self.loss.items():
            if isinstance(key, tuple):
                assert isinstance(outputs, dict), "you specified output key, but output is single tensor"
                label_key, pred_key = key
                label = inputs[label_key]
                prediction = outputs[pred_key]
            else:
                assert isinstance(outputs, tf.Tensor), "got multioutput, but no key was provided"
                label = inputs[key]
                prediction = outputs
            # cast label to be same type as prediction (which is always float32)
            label = tf.cast(label, tf.float32)
            # get sample weights
            sample_weights = inputs.get(self.sample_weight_col, tf.ones_like(prediction))
            # sample weights can be nested to allow for different weights per output / task
            if isinstance(sample_weights, dict):
                sample_weights = sample_weights.get(label_key, tf.ones_like(prediction))
            # masking: NOTE tf ranking losses automatically mask all examples with label=-1!!!
            loss_mask = tf.cast(tf.not_equal(label, -1), tf.float32)
            sample_weights = sample_weights * loss_mask
            #print(sample_weights)
            # calculate loss with given loss function
            loss = loss_fn(label, prediction, sample_weight=sample_weights)
            losses.append(loss)
        # calculate weighted sum of losses
        total_loss = self.loss_weighter(losses, training)
        return total_loss


    def _update_loss_from_function(self, inputs, outputs, training=True):
        assert isinstance(outputs, tf.Tensor), "require single output in function mode"
        sample_weights = inputs.get(self.sample_weight_col, None)
        label = tf.cast(inputs[config.CLICK_COL], tf.float32)
        loss_mask = tf.cast(tf.not_equal(label, -1), tf.float32)
        sample_weights = sample_weights * loss_mask
        loss = self.loss(label, outputs, sample_weight=sample_weights)
        return loss


    def update_loss(self, inputs, outputs, training=True) -> dict:
        if isinstance(self.loss, Callable):
            metric_vals = self._update_loss_from_function(inputs, outputs, training=training)
        elif isinstance(self.loss, dict):
            metric_vals = self._update_loss_from_dict(inputs, outputs, training=training)
        else:
            raise ValueError(f"unknown dtpye {type(self.compiled_metrics._metrics)} for metrics")
        return metric_vals


    def update_metrics(self, inputs, outputs) -> dict:

        if not self.eval_metrics:
            return dict()

        for metric in self.eval_metrics:
            label = inputs[metric.label]
            if not metric.pred:
                assert isinstance(outputs, tf.Tensor), "got multioutput, but no key was provided"
                prediction = outputs
            else:
                assert isinstance(outputs, dict), "you specified output key, but output is single tensor"
                prediction = outputs[metric.pred]

            metric_mask = tf.cast(tf.not_equal(label, -1), tf.float32)
            mask_all_zeros = tf.minimum(1.0, tf.cast(tf.reduce_sum(label, axis=1, keepdims=True), tf.float32))
            final_mask = mask_all_zeros * metric_mask

            metric.metric_fn.update_state(label, prediction, sample_weight=final_mask)

        return {m.metric_fn.name: m.metric_fn.result() for m in self.eval_metrics}


    def _infer_loss_fn(self, **kwargs):
        # NOTE: deprecated. Leave here for backward compatability
        if self.num_negatives is None:
            return binary_crossentropy_loss(**kwargs)
        elif self.num_negatives == 1:
            return triplet_loss(**kwargs)
        elif self.num_negatives > 1:
            return softmax_crossentropy_loss_new(**kwargs)
        elif self.num_negatives == -1:
            return sigmoid_crossentropy_loss(**kwargs)
        else:
            raise ValueError("invalid number of negative samples passed")
            

    @abc.abstractmethod
    def _train_on_batch(self, inputs):
        pass

    @abc.abstractmethod
    def _validate_on_batch(self, inputs):
        pass

    @tf.function
    def _tf_train_on_batch(self, inputs):
        return self._train_on_batch(inputs)

    @tf.function
    def _tf_validate_on_batch(self, inputs):
        return self._validate_on_batch(inputs) 

    def train_step(self, inputs):
        if self.graph_exec:
            return self._tf_train_on_batch(inputs)
        else:
            return self._train_on_batch(inputs)

    def test_step(self, inputs):
        if self.graph_exec:
            return self._tf_validate_on_batch(inputs)
        else:
            return self._validate_on_batch(inputs)   


class SimpleTrainer(BaseTrainer):
    
    def __init__(self,
                 ranker: nn.Model,
                 losses: dict = None,
                 learning_rate: float = 0.001,
                 sample_weight_col: str = None,
                 num_negatives: int = None,
                 graph_exec: bool = True,
                 train_end_to_end: bool = True,
                 loss_weights: list = None,
                 loss_weighting_strategy: str = "random",
                 metrics: List[Metric] = None,
                 clipnorm: float = 2,
                 **kwargs):

        super().__init__(ranker,
                         losses,
                         learning_rate,
                         sample_weight_col,
                         num_negatives,
                         graph_exec,
                         train_end_to_end,
                         loss_weights,
                         loss_weighting_strategy,
                         metrics,
                         clipnorm,
                         **kwargs)


    def call(self, inputs):
        scores = self.ranker(inputs)
        return scores

    def _train_on_batch(self, inputs: dict):
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.update_loss(inputs, outputs)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metric_vals = self.update_metrics(inputs, outputs)
        return {"loss": tf.reduce_mean(loss), **metric_vals}


    def _validate_on_batch(self, inputs):
        outputs = self(inputs, training=False)
        loss = self.update_loss(inputs, outputs)
        metric_vals = self.update_metrics(inputs, outputs)
        return {"loss": loss, **metric_vals}