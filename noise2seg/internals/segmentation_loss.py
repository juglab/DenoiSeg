import keras.backend as K
import tensorflow as tf


def loss_seg(relative_weights):
    """
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.
    """

    class_weights = tf.constant([relative_weights])

    def seg_crossentropy(class_targets, y_pred):
        onehot_labels = tf.reshape(class_targets, [-1, 3])
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)

        a = tf.reduce_sum(onehot_labels, axis=-1)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3]))

        weighted_loss = loss * weights

        return K.mean(a * weighted_loss)

    return seg_crossentropy
