import datetime
import warnings

import numpy as np
import tensorflow as tf
from csbdeep.data import PadAndCropResizer
from csbdeep.internals import nets
from csbdeep.models import CARE
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, save_json, load_json
from csbdeep.utils.six import Path
from keras import backend as K
from keras.callbacks import TerminateOnNaN
from n2v.utils import n2v_utils
from scipy import ndimage
from six import string_types

from noise2seg.models import Noise2SegConfig
from noise2seg.utils.compute_precision_threshold import isnotebook, compute_labels
from ..internals.N2S_DataWrapper import N2S_DataWrapper
from noise2seg.internals.losses import loss_noise2seg, loss_seg
from n2v.utils.n2v_utils import pm_identity, pm_normal_additive, pm_normal_fitted, pm_normal_withoutCP, pm_uniform_withCP
from tqdm import tqdm, tqdm_notebook


class Noise2Seg(CARE):
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
        >>> model = Noise2Seg(config, 'my_model')
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
        config is None or isinstance(config, Noise2SegConfig) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        if config is not None and not config.is_valid():
            invalid_attr = config.is_valid(True)[1]
            raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))
        (not (config is None and basedir is None)) or _raise(ValueError())

        name is None or isinstance(name, string_types) or _raise(ValueError())
        basedir is None or isinstance(basedir, (string_types, Path)) or _raise(ValueError())
        self.config = config
        self.name = name if name is not None else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.basedir = Path(basedir) if basedir is not None else None

        if config is not None:
            # config was provided -> update before it is saved to disk
            self._update_and_check_config()
        self._set_logdir()
        if config is None:
            # config was loaded from disk -> update it after loading
            self._update_and_check_config()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()

    def _build(self):
        return self._build_unet(
            n_dim=self.config.n_dim,
            n_channel_out=self.config.n_channel_out,
            residual=self.config.unet_residual,
            n_depth=self.config.unet_n_depth,
            kern_size=self.config.unet_kern_size,
            n_first=self.config.unet_n_first,
            last_activation=self.config.unet_last_activation,
            batch_norm=self.config.batch_norm
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


    def train(self, X, Y, validation_data, epochs=None, steps_per_epoch=None):
        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.05f%% of all images)" % (100 * frac_val))
        axes = axes_check_and_normalize('S' + self.config.axes, X.ndim)
        ax = axes_dict(axes)
        div_by = 2 ** self.config.unet_n_depth
        axes_relevant = ''.join(a for a in 'XYZT' if a in axes)
        val_num_pix = 1
        train_num_pix = 1
        val_patch_shape = ()
        for a in axes_relevant:
            n = X.shape[ax[a]]
            val_num_pix *= validation_data[0].shape[ax[a]]
            train_num_pix *= X.shape[ax[a]]
            val_patch_shape += tuple([validation_data[0].shape[ax[a]]])
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axes %s"
                    " (axis %s has incompatible size %d)" % (div_by, axes_relevant, a, n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        manipulator = eval('pm_{0}({1})'.format(self.config.n2v_manipulator, str(self.config.n2v_neighborhood_radius)))

        means = np.array([float(mean) for mean in self.config.means], ndmin=len(X.shape), dtype=np.float32)
        stds = np.array([float(std) for std in self.config.stds], ndmin=len(X.shape), dtype=np.float32)

        X = self.__normalize__(X, means, stds)
        validation_X = self.__normalize__(validation_data[0], means, stds)

        # Here we prepare the Noise2Void data. Our input is the noisy data X and as target we take X concatenated with
        # a masking channel. The N2V_DataWrapper will take care of the pixel masking and manipulating.
        training_data = N2S_DataWrapper(X=X,
                                        n2v_Y=np.concatenate((X, np.zeros(X.shape, dtype=X.dtype)), axis=axes.index('C')),
                                        seg_Y=Y,
                                        batch_size=self.config.train_batch_size,
                                        perc_pix=self.config.n2v_perc_pix,
                                        shape=self.config.n2v_patch_shape,
                                        value_manipulation=manipulator)

        # validation_Y is also validation_X plus a concatenated masking channel.
        # To speed things up, we precompute the masking vo the validation data.
        validation_Y = np.concatenate((validation_X, np.zeros(validation_X.shape, dtype=validation_X.dtype)),
                                      axis=axes.index('C'))
        n2v_utils.manipulate_val_data(validation_X, validation_Y,
                                      perc_pix=self.config.n2v_perc_pix,
                                      shape=val_patch_shape,
                                      value_manipulation=manipulator)

        validation_Y = np.concatenate((validation_Y, validation_data[1]), axis=-1)

        history = self.keras_model.fit_generator(generator=training_data, validation_data=(validation_X, validation_Y),
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)

        if self.basedir is not None:
            self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

            if self.config.train_checkpoint is not None:
                print()
                self._find_and_load_weights(self.config.train_checkpoint)
                try:
                    # remove temporary weights
                    (self.logdir / 'weights_now.h5').unlink()
                except FileNotFoundError:
                    pass

        return history


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

    def predict_label_masks(self, X, Y, threshold, measure):
        predicted_images = []
        precision_result = []
        for i in range(X.shape[0]):
            if( np.max(Y[i])==0 and np.min(Y[i])==0 ):
                continue
            else:
                prediction = self.predict(X[i].astype(np.float32), axes='YX')
                labels = compute_labels(prediction, threshold)
                tmp_score = measure(Y[i], labels)
                predicted_images.append(labels)
                precision_result.append(tmp_score)
        return predicted_images, np.mean(precision_result)

    def optimize_thresholds(self, X_val, Y_val, measure):
        """
         Computes average precision (AP) at different probability thresholds on validation data and returns the best-performing threshold.

         Parameters
         ----------
         X_val : array(float)
             Array of validation images.
         Y_val : array(float)
             Array of validation labels
         model: keras model

         mode: 'none', 'StarDist'
             If `none`, consider a U-net type model, else, considers a `StarDist` type model
         Returns
         -------
         computed_threshold: float
             Best-performing threshold that gives the highest AP.


         """
        print('Computing best threshold: ')
        precision_scores = []
        if (isnotebook()):
            progress_bar = tqdm_notebook
        else:
            progress_bar = tqdm
        for ts in progress_bar(np.linspace(0.1, 1, 19)):
            _, score = self.predict_label_masks(X_val, Y_val, ts, measure)
            precision_scores.append((ts, score))
            print('Score for threshold =', "{:.2f}".format(ts), 'is', "{:.4f}".format(score))

        sorted_score = sorted(precision_scores, key=lambda tup: tup[1])[-1]
        computed_threshold = sorted_score[0]
        best_score = sorted_score[1]
        return computed_threshold, best_score

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
        means = np.array([float(mean) for mean in self.config.means], ndmin=len(img.shape), dtype=np.float32)
        stds = np.array([float(std) for std in self.config.stds], ndmin=len(img.shape), dtype=np.float32)

        if img.dtype != np.float32:
            print('The input image is of type {} and will be casted to float32 for prediction.'.format(img.dtype))
            img = img.astype(np.float32)

        # new_axes = axes
        new_axes = axes
        if 'C' in axes:
            new_axes = axes.replace('C', '') + 'C'
            normalized = self.__normalize__(np.moveaxis(img, axes.index('C'), -1), means, stds)
        else:
            normalized = self.__normalize__(img[..., np.newaxis], means, stds)
            normalized = normalized[..., 0]

        pred_full = self._predict_mean_and_scale(normalized, axes=new_axes, normalizer=None, resizer=resizer, n_tiles=n_tiles)[0]

        pred_denoised = self.__denormalize__(pred_full[...,:1], means, stds)
        
        pred = np.concatenate([pred_denoised, pred_full[...,1:]], axis=-1)
        
        if 'C' in axes:
            pred = np.moveaxis(pred, -1, axes.index('C'))

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

        if self.config.train_loss == 'seg':
            loss_standard = eval('loss_seg(relative_weights=%s)' % self.config.relative_weights)
        elif self.config.train_loss == 'noise2seg':
            loss_standard = eval('loss_noise2seg(weight_denoise={}, relative_weights={})'.format(
                self.config.n2s_weight_denoise,
                self.config.relative_weights))
        else:
            _raise('Unknown Loss!')

        _metrics = [loss_standard]
        callbacks = [TerminateOnNaN()]

        # compile model
        model.compile(optimizer=optimizer, loss=loss_standard, metrics=_metrics)

        return callbacks

    def __normalize__(self, data, means, stds):
        return (data - means) / stds


    def __denormalize__(self, data, means, stds):
        return (data * stds) + means

    def _set_logdir(self):
        self.logdir = self.basedir / self.name

        config_file = self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = self._config_class(np.array([]), **config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn(
                    'output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))

    @property
    def _config_class(self):
        return Noise2SegConfig
