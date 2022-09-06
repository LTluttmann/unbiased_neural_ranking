import tensorflow as tf
import tensorflow.keras as nn

from mt.models.learning_algorithms import BaseTrainer
from mt.models.layers import create_tower
from mt.config import config

from typing import Callable



class JointEstimator(BaseTrainer):

    def __init__(self,
                 ranker: nn.Model,
                 losses: dict = None,
                 learning_rate: float = 0.001,
                 sample_weight_col: str = None,
                 num_negatives: int = None,
                 graph_exec: bool = True,
                 train_end_to_end: bool = True,
                 clipnorm: float = 2.0,
                 joe_nodes: list = [1],
                 joe_activations: str = None,
                 joe_output_activation: str = "sigmoid",
                 joe_dropout: float = None,
                 joe_batch_norm: bool = True,
                 joe_input_batch_norm: bool = True,
                 joe_multiplicative: bool = True,
                 score_weights: list = None,
                 loss_weights: list = None,
                 loss_weighting_strategy: str= "random",
                 **kwargs):

        super().__init__(ranker=ranker,
                         losses=losses,
                         sample_weight_col=sample_weight_col,
                         learning_rate=learning_rate,
                         num_negatives=num_negatives,
                         graph_exec=graph_exec,
                         train_end_to_end=train_end_to_end,
                         clipnorm=clipnorm,
                         loss_weights=loss_weights,
                         loss_weighting_strategy=loss_weighting_strategy,
                         **kwargs)

        self.propensity_estimator = create_tower(hidden_layer_dims=joe_nodes,
                                                 output_units=1, 
                                                 activation=joe_activations, 
                                                 output_activation=joe_output_activation, 
                                                 input_batch_norm=joe_input_batch_norm, 
                                                 use_batch_norm=joe_batch_norm, 
                                                 dropout=joe_dropout,
                                                 name="bias_model")

        self.interaction_fn: Callable = tf.multiply if joe_multiplicative else tf.add
        
        self.score_weights = score_weights


    def _get_ranker_input(self, inputs):
        if hasattr(self.ranker, "input_signature"):
            ranker_inputs = {k: v for k,v in inputs.items() if k in self.ranker.input_signature.keys()}
            return ranker_inputs
        else:
            return inputs


    def call(self, inputs, training=True):
        inputs = inputs.copy()
        # [BS, PN, F]
        pb_features = inputs.pop(config.POS_BIAS_FEATURE_COL)
        batch_size = tf.shape(pb_features)[0]

        # [BS, PN, 1]
        propens = self.propensity_estimator(pb_features, training=training)   
        propens = tf.reshape(propens, (batch_size, -1))
        # [BS, PN, 1]
        ranker_inputs = self._get_ranker_input(inputs)
        relevance_scores = self.ranker(ranker_inputs, training=training)
        relevance_scores = [relevance_scores] if not isinstance(relevance_scores, list) else relevance_scores
        
        if self.score_weights is None:
            self.score_weights = [1] * len(relevance_scores)
        else:
            assert len(self.score_weights) == len(relevance_scores)

        combined_score = tf.add_n([self.score_weights[i] * relevance_scores[i] for i in range(len(relevance_scores))])
        return_values = {}
        for i, score in enumerate(relevance_scores):
            label_name = config.LABELS[i]
            prob = self.interaction_fn(score, propens)
            return_values[f"{label_name}_probs"] = prob
            return_values[f"{label_name}_scores"] = score
        return_values["combined_scores"] = combined_score
        
        return return_values

    def _train_on_batch(self, inputs: dict):
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss = self.update_loss(inputs, outputs, training=True)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metric_vals = self.update_metrics(inputs, outputs)
        return {"loss": tf.reduce_mean(loss), **metric_vals}


    def _validate_on_batch(self, inputs):
        outputs = self(inputs, training=False)
        loss = self.update_loss(inputs, outputs, training=False)
        metric_vals = self.update_metrics(inputs, outputs)
        return {"loss": tf.reduce_mean(loss), **metric_vals}
