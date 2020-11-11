import keras.backend as K
import tensorflow as tf
from n2v.internals.n2v_losses import loss_mse as n2v_loss
from tensorflow.nn import softmax_cross_entropy_with_logits_v2 as cross_entropy


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


def denoiseg_seg_loss(relative_weights=[1.0, 1.0, 5.0]):
    class_weights = tf.constant([relative_weights])

    def seg_loss(y_true, y_pred):
        channel_axis = len(y_true.shape) - 1
        bg, fg, b = tf.split(y_true, 3, axis=channel_axis)
        pred_bg, pred_fg, pred_b = tf.split(y_pred, 3, axis=len(y_pred.shape) - 1)

        onehot_gt = tf.reshape(tf.stack([bg, fg, b], axis=3), [-1, 3])
        weighted_gt = tf.reduce_sum(class_weights * onehot_gt, axis=1)

        onehot_pred = tf.reshape(tf.stack([pred_bg, pred_fg, pred_b], axis=channel_axis), [-1, 3])

        segmentation_loss = K.mean(
            tf.reduce_sum(onehot_gt, axis=-1) * (cross_entropy(logits=onehot_pred, labels=onehot_gt) * weighted_gt)
        )

        return segmentation_loss

    return seg_loss


def denoiseg_denoise_loss():
    n2v_mse_loss = n2v_loss()

    def denoise_loss(y_true, y_pred):
        channel_axis = len(y_true.shape) - 1
        target, mask = tf.split(y_true, 2, axis=channel_axis)
        return n2v_mse_loss(tf.concat([target, mask], axis=channel_axis), y_pred)

    return denoise_loss
