import datetime

import numpy as np
import tensorflow as tf
from csbdeep.data import PadAndCropResizer
from csbdeep.internals import nets
from csbdeep.models import CARE
from csbdeep.utils import _raise
from csbdeep.utils.six import Path
from keras import backend as K
from keras.callbacks import TerminateOnNaN
from scipy import ndimage
from six import string_types

from noise2seg.models import SegConfig
from noise2seg.utils.compute_precision_threshold import compute_threshold, precision
from noise2seg.internals.segmentation_loss import loss_seg


class Seg(CARE):
    """The training scheme to train a standard 3-class segmentation network.
        Uses a convolutional neural network created by :func:`csbdeep.internals.nets.custom_unet`.
        Parameters
        ----------
        config : :class:`voidseg.models.seg_config` or None
            Valid configuration of Seg network (see :func:`SegConfig.is_valid`).
            Will be saved to disk as JSON (``config.json``).
            If set to ``None``, will be loaded from disk (must exist).
        name : str or None
            Model name. Uses a timestamp if set to ``None`` (default).
        basedir : str
            Directory that contains (or will contain) a folder with the given model name.
            Use ``None`` to disable saving (or loading) any data to (or from) disk (regardless of other parameters).
        Raises
        ------
        FileNotFoundError
            If ``config=None`` and config cannot be loaded from disk.
        ValueError
            Illegal arguments, including invalid configuration.
        Example
        -------
        >>> model = Seg(config, 'my_model')
        Attributes
        ----------
        config : :class:`voidseg.models.seg_config`
            Configuration of Seg trainable CARE network, as provided during instantiation.
        keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
            Keras neural network model.
        name : str
            Model name.
        logdir : :class:`pathlib.Path`
            Path to model folder (which stores configuration, weights, etc.)
        """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring"""
        config is None or isinstance(config, SegConfig) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(ValueError())

        name is None or isinstance(name, string_types) or _raise(ValueError())
        basedir is None or isinstance(basedir, (string_types, Path)) or _raise(ValueError())
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()
        else:
            config.probabilistic = False

    def _build(self):
        return self._build_unet(
            n_dim            = self.config.n_dim,
            n_channel_out    = self.config.n_channel_out,
            residual         = self.config.unet_residual,
            n_depth          = self.config.unet_n_depth,
            kern_size        = self.config.unet_kern_size,
            n_first          = self.config.unet_n_first,
            last_activation  = self.config.unet_last_activation,
            batch_norm       = self.config.batch_norm
        )(self.config.unet_input_shape)

    def _build_unet(self, n_dim=2, n_depth=2, kern_size=3, n_first=32, n_channel_out=1, residual=False,
                    last_activation='linear', batch_norm=True):
        """Construct a common CARE neural net based on U-Net [1]_ to be used for image segmentation.
           Parameters
           ----------
           n_dim : int
               number of image dimensions (2 or 3)
           n_depth : int
               number of resolution levels of U-Net architecture
           kern_size : int
               size of convolution filter in all image dimensions
           n_first : int
               number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
           n_channel_out : int
               number of channels of the predicted output image
           last_activation : str
               name of activation function for the final output layer
           batch_norm : bool
               Use batch normalization during training
           Returns
           -------
           function
               Function to construct the network, which takes as argument the shape of the input image
           Example
           -------
           >>> model = common_unet(2, 2, 3, 32, 1, False, 'linear', False)(input_shape)
           References
           ----------
           .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
           """

        def _build_this(input_shape):
            return nets.custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,) * n_dim,
                                    pool_size=(2,) * n_dim, n_channel_out=n_channel_out, residual=residual,
                                    prob_out=False, batch_norm=batch_norm)

        return _build_this

    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.
        Calls :func:`prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.
        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.
        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`prepare_model`.
        """
        if optimizer is None:
            from keras.optimizers import Adam
            optimizer = Adam(lr=self.config.train_learning_rate)

        # TODO: This line is the reason for the existence of this method.
        # TODO: CARE calls prepare_model from train, but we have to overwrite prepare_model.
        self.callbacks = self.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            if self.config.train_checkpoint is not None:
                from keras.callbacks import ModelCheckpoint
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True,
                                    save_weights_only=True))
                self.callbacks.append(
                    ModelCheckpoint(str(self.logdir / 'weights_now.h5'), save_best_only=False, save_weights_only=True))

            if self.config.train_tensorboard:
                from csbdeep.utils.tf import CARETensorBoard

                class SegTensorBoard(CARETensorBoard):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}

                        if self.validation_data and self.freq:
                            if epoch % self.freq == 0:
                                # TODO: implement batched calls to sess.run
                                # (current call will likely go OOM on GPU)
                                tensors = self.model.inputs + self.gt_outputs + self.model.sample_weights
                                if self.model.uses_learning_phase:
                                    tensors += [K.learning_phase()]
                                    val_data = list(v[:self.n_images] for v in self.validation_data[:-1])
                                    val_data += self.validation_data[-1:]
                                else:
                                    val_data = list(v[:self.n_images] for v in self.validation_data)
                                # GIT issue 20: We need to remove the masking component from the validation data to prevent crash.
                                end_index = (val_data[1].shape)[-1] // 2
                                val_data[1] = val_data[1][..., :end_index]
                                feed_dict = dict(zip(tensors, val_data))
                                result = self.sess.run([self.merged], feed_dict=feed_dict)
                                summary_str = result[0]

                                self.writer.add_summary(summary_str, epoch)

                        for name, value in logs.items():
                            if name in ['batch', 'size']:
                                continue
                            summary = tf.Summary()
                            summary_value = summary.value.add()
                            summary_value.simple_value = value.item()
                            summary_value.tag = name
                            self.writer.add_summary(summary, epoch)

                        self.writer.flush()

                self.callbacks.append(
                    SegTensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True,
                                   prob_out=self.config.probabilistic))

        if self.config.train_reduce_lr is not None:
            from keras.callbacks import ReduceLROnPlateau
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def predict_label_masks(self, X, Y, threshold):
        predicted_images = []
        precision_result = []
        for i in range(X.shape[0]):
            pred_ = self.predict(X[i].astype(np.float32), axes='YX')
            prediction_exp = np.exp(pred_[..., :])
            prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
            prediction_fg = prediction_seg[..., 1]
            pred_thresholded = prediction_fg > threshold
            labels, nb = ndimage.label(pred_thresholded)
            predicted_images.append(labels)
            precision_result.append(precision(Y[i], predicted_images[i]))
        return predicted_images, np.mean(precision_result)

    def optimize_thresholds(self, valdata, valmasks):
        return compute_threshold(valdata, valmasks, self)

    def predict(self, img, axes, resizer=PadAndCropResizer(), n_tiles=None):
        """
        Apply the network to so far unseen data. 
        Parameters
        ----------
        img     : array(floats) of images
        axes    : String
                  Axes of the image ('YX').
        resizer : class(Resizer), optional(default=PadAndCropResizer())
        n_tiles : tuple(int)
                  Number of tiles to tile the image into, if it is too large for memory.
        Returns
        -------
        image : array(float)
                The restored image.
        """

        if img.dtype != np.float32:
            print('The input image is of type {} and will be casted to float32 for prediction.'.format(img.dtype))
            img = img.astype(np.float32)

        new_axes = axes
        normalized = img[..., np.newaxis]
        normalized = normalized[..., 0]
        pred = \
        self._predict_mean_and_scale(normalized, axes=new_axes, normalizer=None, resizer=resizer, n_tiles=n_tiles)[0]

        return pred

    def prepare_model(self, model, optimizer, loss):
        """
         Called by `prepare_for_training` function.
         Parameters
        ----------
        model : Seg object.

        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        loss: `loss_seg`
            computes Cross-Entropy between the class targets and predicted outputs

        Returns
        ----------
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        """

        from keras.optimizers import Optimizer
        isinstance(optimizer, Optimizer) or _raise(ValueError())

        loss_standard = eval('loss_seg(relative_weights=%s)' % self.config.relative_weights)

        _metrics = [loss_standard]
        callbacks = [TerminateOnNaN()]

        # compile model
        model.compile(optimizer=optimizer, loss=loss_standard, metrics=_metrics)

        return callbacks
