import tensorflow as tf
import tensorflow.keras as nn

from mt.models.learning_algorithms import BaseTrainer
from mt.models.layers import create_tower
from mt.config import config

from typing import Callable, List
from collections import defaultdict
from tqdm import tqdm


class EMAlgorithm():
    
    def __init__(self, relevance_prior, examination_prior) -> None:
        self.relevance_prior = relevance_prior
        self.examination_prior = examination_prior

    @staticmethod
    def update_vals_in_m_step(n:dict, d: dict, param: dict):

        for k, v in d.items():
            if isinstance(v, dict):
                for k2, _ in v.items():
                    num = n[k][k2]
                    den = d[k][k2]
                    param[k][k2] = num/den
            else:
                num = n[k]
                den = d[k]
                param[k] = num/den
        return param
        
    def __call__(self, dataset, iterations):
        beta = defaultdict(lambda: self.examination_prior)
        alpha = defaultdict(lambda: defaultdict(lambda: self.relevance_prior))

        numerator_beta = defaultdict(lambda: 0)
        numerator_alpha = defaultdict(lambda: defaultdict(lambda: 0)) 
        
        denominator_beta = defaultdict(lambda: 0)
        denominator_alpha = defaultdict(lambda: defaultdict(lambda: 0))

        for _ in tqdm(range(iterations)):
            
            # E-step
            for x  in iter(dataset):
                qid = x["id"].numpy()[0].decode("utf-8")
                
                clicks = x["click"].numpy()

                pid = x["pid"].numpy()
                
                for k, (c, p) in enumerate(zip(clicks, pid)):
                    
                    # return qid, c, p

                    beta_k = beta[k]
                    alpha_q_d = alpha[qid][p]
                    
                    p_e1_r0_c0 = (beta_k * (1 - alpha_q_d)) / (1 - beta_k * alpha_q_d)
                    # prob of relevant, but not examined // [BS, 1]
                    p_e0_r1_c0 = ((1 - beta_k) * alpha_q_d) / (1 - beta_k * alpha_q_d)

        
                    numerator_beta[k] += (c + (1-c) * p_e1_r0_c0)
                    denominator_beta[k] += 1

                    numerator_alpha[qid][p] += (c + (1 - c) * p_e0_r1_c0)
                    denominator_alpha[qid][p] += 1

            # M-step
            alpha = self.update_vals_in_m_step(numerator_alpha, denominator_alpha, alpha)
            beta = self.update_vals_in_m_step(numerator_beta, denominator_beta, beta)
        return alpha, beta


class IPWEstimator(BaseTrainer):

    def __init__(self,
                 ranker: nn.Model,
                 propensities: List[int],
                 losses: dict = None,
                 learning_rate: float = 0.001,
                 sample_weight_col: str = None,
                 num_negatives: int = None,
                 graph_exec: bool = True,
                 train_end_to_end: bool = True,
                 clipnorm: float = 2.0,
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
        
        self.propensities = propensities
        self.score_weights = score_weights


    def _get_ranker_input(self, inputs):
        if hasattr(self.ranker, "input_signature"):
            ranker_inputs = {k: v for k,v in inputs.items() if k in self.ranker.input_signature.keys()}
            return ranker_inputs
        else:
            return inputs


    def call(self, inputs, training=True):
        inputs = inputs.copy()

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

    def _update_loss_from_dict(self, inputs, outputs, training=True):
        losses = []
        # inverse propensity values
        ipw=tf.math.divide_no_nan(1.0, tf.gather(self.propensities, inputs["actual_position"], batch_dims=1, axis=0))

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
            sample_weights = sample_weights * loss_mask * ipw
            #print(sample_weights)
            # calculate loss with given loss function
            loss = loss_fn(label, prediction, sample_weight=sample_weights)
            losses.append(loss)
        # calculate weighted sum of losses
        total_loss = self.loss_weighter(losses, training)
        return total_loss

    def _update_loss_from_function(self, inputs, outputs, training=True):
        assert isinstance(outputs, tf.Tensor), "require single output in function mode"
        ipw=tf.math.divide_no_nan(1.0, tf.gather(self.propensities, inputs["position"], batch_dims=1, axis=0))
        sample_weights = inputs.get(self.sample_weight_col, None)
        label = tf.cast(inputs[config.CLICK_COL], tf.float32)
        loss_mask = tf.cast(tf.not_equal(label, -1), tf.float32)
        sample_weights = sample_weights * loss_mask * ipw
        loss = self.loss(label, outputs, sample_weight=sample_weights)
        return loss