
import tensorflow as tf
import tensorflow.keras as nn

from mt.models.learning_algorithms import BaseTrainer
from mt.config import config


def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.
        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.
        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.
        """
    return tf.math.ceil(probs - tf.random.uniform(tf.shape(probs)))


class RegressionEM(BaseTrainer):
    def __init__(
            self, 
            ranker: nn.Model, 
            rank_list_size: int,
            em_step_size: float = 0.05,
            learning_rate: float = 0.001, 
            beta_1: float = 0.9, 
            beta_2: float = 0.999,
            graph_exec: bool = True, 
            train_end_to_end: bool = True, 
            **kwargs):
        # only allows pointwise estimation, thus set num_negatives to None
        super().__init__(
            ranker=ranker, 
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            num_negatives=None, 
            graph_exec=graph_exec, 
            train_end_to_end=train_end_to_end, 
            **kwargs)

        # LS
        self.rank_list_size = rank_list_size
        # [1, LS]
        self.propensity = tf.Variable(tf.ones((1, self.rank_list_size)) * 0.9, trainable=False)
        self.em_step_size = em_step_size

    def call(self, inputs, training=None, mask=None):

        relevance_scores = self.ranker(inputs, training=training)

        return relevance_scores

    def _train_on_batch(self, inputs):
        x = inputs
        train_labels = tf.cast(x[config.CLICK_COL], tf.float32)
        # [BS, 1]
        train_labels = tf.reshape(train_labels, (-1, 1))
        # [BS, LS]
        # TODO etwas verwirrend, dass hier der one-hot vector übergeben wird
        positonal_mask = x["position"]
        # [BS]
        position = tf.argmax(positonal_mask, axis=1)
        with tf.GradientTape() as tape:
            # [BS, 1]
            relevance_scores = self(x, training=True)
            # relevance_scores = tf.squeeze(relevance_scores)
            with tape.stop_recording():
                # Conduct estimation step. Calculations not relevant for model gradients
                # [BS, 1]
                gamma = tf.sigmoid(relevance_scores)
                # [1, LS] -> [1, BS] -> [BS, 1]
                propensities_per_pos = tf.transpose(
                    tf.gather(self.propensity, position, axis=1))
                # probability of clicking when observed and relevant // [BS, 1]
                p_e1_r1_c1 = 1
                # prob of not clicking when examined (i.e. examined but not relevant) // [BS, 1]
                p_e1_r0_c0 = propensities_per_pos * \
                    (1 - gamma) / (1 - propensities_per_pos * gamma)
                # prob of relevant, but not examined // [BS, 1]
                p_e0_r1_c0 = (1 - propensities_per_pos) * gamma / \
                    (1 - propensities_per_pos * gamma)
                # prob of neither relevant nor examined // [BS, 1]
                p_e0_r0_c0 = (1 - propensities_per_pos) * \
                    (1 - gamma) / (1 - propensities_per_pos * gamma)
                # probability of examination // [BS, 1]
                p_e1 = p_e1_r0_c0 + p_e1_r1_c1
                # probability of being relevant // [BS,1]
                p_r1 = train_labels + (1 - train_labels) * p_e0_r1_c0
                # update values for propensity // [BS, 1]
                update_vals = train_labels + (1-train_labels) * p_e1_r0_c0
                # we might have several observations for the same position. We have to average those before
                # performing the actual update step. We first sum all update values per position...
                # [BS, 1] * [BS, LS] -> [BS, LS] -> reduce to: [1, LS]
                sum_update_positionwise = tf.reduce_sum(
                    update_vals * positonal_mask, axis=0, keepdims=True)
                # ...and then divide by the number of observations per position
                mean_update_positionwise = sum_update_positionwise / \
                    tf.maximum(tf.reduce_sum(
                        positonal_mask, axis=0, keepdims=True), 1)

                # for the update step, those positions that have not been in batch must be updated with old value
                # [1, LS] * [1, LS] -> [1, LS]
                old_vals_of_non_observed_positions = self.propensity * \
                    (1-tf.reduce_max(positonal_mask, axis=0, keepdims=True))

                tf.assert_equal(tf.boolean_mask(mean_update_positionwise, tf.cast(
                    1-tf.reduce_max(positonal_mask, 0, keepdims=True), tf.bool)), 0.0)
                tf.assert_equal(tf.boolean_mask(old_vals_of_non_observed_positions, tf.cast(
                    tf.reduce_max(positonal_mask, 0, keepdims=True), tf.bool)), 0.0)

                # [1, LS]
                final_update_vals = mean_update_positionwise + old_vals_of_non_observed_positions
                new_propensity = (
                    1 - self.em_step_size) * self.propensity + self.em_step_size * final_update_vals
                self.propensity.assign(new_propensity)
                ranker_labels = get_bernoulli_sample(p_r1)
            # loss calculation
            loss = self.loss_fn(y_true=ranker_labels, y_pred=relevance_scores)
        # update ranker
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": tf.reduce_mean(loss)}

    def _validate_on_batch(self, inputs):
        inputs = inputs.copy()
        train_labels = inputs.pop(config.CLICK_COL)
        train_labels = tf.cast(train_labels, tf.float32)
        relevance_scores = self(inputs, training=True)
        loss = self.loss_fn(y_true=train_labels, y_pred=relevance_scores)
        metric_vals = self.update_metrics(inputs, relevance_scores)
        return {
            "loss": tf.reduce_mean(loss), 
            **metric_vals
        }

