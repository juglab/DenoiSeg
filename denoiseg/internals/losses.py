import tensorflow.keras.backend as K
import tensorflow as tf
from n2v.internals.n2v_losses import loss_mse as n2v_loss
from tensorflow.nn import softmax_cross_entropy_with_logits as cross_entropy


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

        loss = tf.nn.cross_entropy(labels=onehot_labels, logits=tf.reshape(y_pred, [-1, 3]))

        weighted_loss = loss * weights

        return K.mean(a * weighted_loss)

    return seg_crossentropy


def loss_denoiseg(alpha=0.5, relative_weights=[1.0, 1.0, 5.0], n_chan=1):
    """
    Calculate DenoiSeg loss which is a weighted sum of segmentation- and
    noise2void-loss

    :param lambda_: relative weighting, 0 means denoising, 1 means segmentation; (Default: 0.5)
    :param relative_weights: Segmentation class weights (background, foreground, border); (Default: [1.0, 1.0, 5.0])
    :return: DenoiSeg loss
    """
    denoise_loss = denoiseg_denoise_loss(weight=alpha, n_chan=n_chan)
    seg_loss = denoiseg_seg_loss(weight=(1 - alpha), relative_weights=relative_weights, n_chan=n_chan)

    def denoiseg(y_true, y_pred):
        return seg_loss(y_true, y_pred) + denoise_loss(y_true, y_pred)

    return denoiseg


def denoiseg_seg_loss(weight=0.5, relative_weights=[1.0, 1.0, 5.0], n_chan=1):
    class_weights = tf.constant([relative_weights])

    def seg_loss(y_true, y_pred):
        targets, masks, bg, fg, b = split_y_true(y_true, n_chan)
        denoiseds, pred_bg, pred_fg, pred_b = split_y_pred(y_pred, n_chan)
        assert len(denoiseds) == len(targets) == len(masks)

        onehot_gt = tf.reshape(tf.stack([bg, fg, b], axis=-1), [-1, 3])
        weighted_gt = tf.reduce_sum(class_weights * onehot_gt, axis=1)
        onehot_pred = tf.reshape(tf.stack([pred_bg, pred_fg, pred_b], axis=channel_axis), [-1, 3])
        segmentation_loss = K.mean(tf.reduce_sum(onehot_gt, axis=-1) * (cross_entropy(logits=onehot_pred, labels=onehot_gt) * weighted_gt))
        return weight * segmentation_loss

    return seg_loss


def denoiseg_denoise_loss(weight=0.5, n_chan=1):
    n2v_mse_loss = n2v_loss()

    def denoise_loss(y_true, y_pred):

        targets, masks, bg, fb, b = split_y_true(y_true, n_chan)
        denoiseds, pred_bg, pred_fg, pred_b = split_y_pred(y_pred, n_chan)

        assert len(denoiseds) == len(targets) == len(masks)

        losses = [
            weight * n2v_mse_loss(tf.concat([target, mask], axis=len(y_true.shape) - 1), denoised)
            for target, mask, denoised in zip(targets, masks, denoiseds)
        ]

        return sum(losses)

    return denoise_loss

def split_y_true(y_true, n_chan):

    channel_axis = len(y_true.shape) - 1
    # 2 outputs per input channel (target and mask), plus FG, BG, boundary
    splits = tf.split(y_true, n_chan * 2 + 3, axis=channel_axis)
    bg, fg, b = splits[-3:]
    n_chan = (len(splits) - 3) // 2
    targets = splits[:n_chan]
    masks = splits[n_chan:2 * n_chan]

    return targets, masks, bg, fg, b

def split_y_pred(y_pred, n_chan):

    # one output per input channel (denoised), plus FG, BG, boundary
    splits = tf.split(y_pred, n_chan + 3, axis=len(y_pred.shape) - 1)
    pred_bg, pred_fg, pred_b = splits[-3:]
    denoiseds = splits[:-3]

    return denoiseds, pred_bg, pred_fg, pred_b
