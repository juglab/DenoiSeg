import datetime
import os
import shutil
import warnings
from zipfile import ZipFile

import keras
import numpy as np
import tensorflow as tf
from csbdeep.data import PadAndCropResizer
from csbdeep.internals import predict
from csbdeep.internals.predict import Progress, to_tensor, from_tensor, tile_iterator_1d
from csbdeep.models import CARE
from csbdeep.models.base_model import suppress_without_basedir
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict, save_json, load_json
from csbdeep.utils.six import Path
from csbdeep.utils.six import tempfile
from csbdeep.version import __version__ as package_version
from keras import backend as K
from keras.callbacks import TerminateOnNaN
from n2v.utils import n2v_utils
from ruamel.yaml import YAML
from six import string_types
from tifffile import imsave
from denoiseg.internals.losses import denoiseg_denoise_loss, denoiseg_seg_loss
from n2v.utils.n2v_utils import pm_identity, pm_normal_additive, pm_normal_fitted, pm_normal_withoutCP, pm_uniform_withCP
from tqdm import tqdm, tqdm_notebook

from denoiseg.internals.losses import denoiseg_denoise_loss, denoiseg_seg_loss
from denoiseg.models import DenoiSegConfig
from denoiseg.utils.compute_precision_threshold import isnotebook, compute_labels
from ..internals.DenoiSeg_DataWrapper import DenoiSeg_DataWrapper
from ..nets.denoiseg_net import denoiseg_model


class DenoiSeg(CARE):
    """The training scheme to train a standard 3-class segmentation network.
        Uses a convolutional neural network created by :func:`csbdeep.internals.nets.custom_unet`.
        Parameters
        ----------
        config : :class:`denoiseg.models.denoiseg_config` or None
            Valid configuration of Seg network (see :func:`denoiseg_config.is_valid`).
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
        >>> model = DenoiSeg(config, 'my_model')
        Attributes
        ----------
        config : :class:`denoiseg.models.denoiseg_config`
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
        config is None or isinstance(config, DenoiSegConfig) or _raise(ValueError('Invalid configuration: %s' % str(config)))
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

        self.alpha = K.variable(value=1, dtype='float32')

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
            return denoiseg_model(input_shape, last_activation, n_depth, n_first, (kern_size,) * n_dim,
                                    pool_size=(2,) * n_dim, n_channel_out=n_channel_out, residual=residual,
                                    batch_norm=batch_norm)

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
        training_data = DenoiSeg_DataWrapper(X=X,
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

        # validation_Y = np.concatenate((validation_Y, validation_data[1]), axis=-1)

        history = self.keras_model.fit_generator(generator=training_data,
                                                 validation_data=(validation_X,
                                                                  {'out_denoise': validation_Y,
                                                                   'out_segment': validation_data[1]}
                                                                  ),
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
                    def set_model(self, model):
                        self.model = model
                        self.sess = K.get_session()
                        tf_sums = []

                        if self.compute_histograms and self.freq and self.merged is None:
                            for layer in self.model.layers:
                                for weight in layer.weights:
                                    tf_sums.append(tf.compat.v1.summary.histogram(weight.name, weight))

                                if hasattr(layer, 'output'):
                                    tf_sums.append(tf.compat.v1.summary.histogram('{}_out'.format(layer.name),
                                                                        layer.output))

                        def _gt_shape(output_shape):
                            return list(output_shape[:-1]) + [1]

                        self.gt_outputs = [K.placeholder(shape=_gt_shape(K.int_shape(x))) for x in self.model.outputs]

                        n_inputs, n_outputs = len(self.model.inputs), len(self.model.outputs)
                        image_for_inputs = np.arange(
                            n_inputs) if self.image_for_inputs is None else self.image_for_inputs
                        image_for_outputs = np.arange(
                            n_outputs) if self.image_for_outputs is None else self.image_for_outputs

                        input_slices = (slice(None),) if self.input_slices is None else self.input_slices
                        output_slices = (slice(None),) if self.output_slices is None else self.output_slices
                        if isinstance(input_slices[0], slice):  # apply same slices to all inputs
                            input_slices = [input_slices] * len(image_for_inputs)
                        if isinstance(output_slices[0], slice):  # apply same slices to all outputs
                            output_slices = [output_slices] * len(image_for_outputs)
                        len(input_slices) == len(image_for_inputs) or _raise(ValueError())
                        len(output_slices) == len(image_for_outputs) or _raise(ValueError())

                        def _name(prefix, layer, i, n, show_layer_names=False):
                            return '{prefix}{i}{name}'.format(
                                prefix=prefix,
                                i=(i if n > 1 else ''),
                                name='' if (layer is None or not show_layer_names) else '_' + ''.join(
                                    layer.name.split(':')[:-1]),
                            )

                        # inputs
                        for i, sl in zip(image_for_inputs, input_slices):
                            layer_name = _name('net_input', self.model.inputs[i], i, n_inputs)
                            input_layer = self.model.inputs[i][tuple(sl)]
                            tf_sums.append(tf.compat.v1.summary.image(layer_name, input_layer, max_outputs=self.n_images))

                        # outputs
                        for i, sl in zip(image_for_outputs, output_slices):
                            # target
                            output_layer = self.gt_outputs[i][tuple(sl)]
                            layer_name = _name('net_target', self.model.outputs[i], i, n_outputs)
                            tf_sums.append(tf.summary.image(layer_name, output_layer, max_outputs=self.n_images))
                            # prediction
                            sep = 1
                            output_layer = self.model.outputs[i][..., :sep][tuple(sl)]
                            layer_name = _name('net_output', self.model.outputs[i], i, n_outputs)
                            tf_sums.append(
                                tf.summary.image(layer_name, output_layer, max_outputs=self.n_images))

                        with tf.name_scope('merged'):
                            self.merged = tf.compat.v1.summary.merge(tf_sums)

                        with tf.name_scope('summary_writer'):
                            if self.write_graph:
                                self.writer = tf.compat.v1.summary.FileWriter(self.log_dir,
                                                                    self.sess.graph)
                            else:
                                self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

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
                                val_data[1] = val_data[1][..., :1]
                                val_data[2] = val_data[2][..., :1]
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
                denoised, segmentation = self.predict(X[i].astype(np.float32), axes='YX')
                labels = compute_labels(segmentation, threshold)
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

        pred_denoised, pred_segment = self.predict_mean_and_scale(normalized, axes=new_axes, resizer=resizer, n_tiles=n_tiles)

        pred_denoised = self.__denormalize__(pred_denoised, means, stds)
        
        if 'C' in axes:
            pred_denoised = np.moveaxis(pred_denoised, -1, axes.index('C'))
            pred_segment = np.moveaxis(pred_segment, -1, axes.index('C'))

        return pred_denoised, pred_segment

    def predict_mean_and_scale(self, img, axes, resizer, n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        _, resizer = self._check_normalizer_resizer(None, resizer)
        # axes = axes_check_and_normalize(axes,img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes,img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C','') # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        x = _permute_axes(img)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes) # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == x.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)
        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n,undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n,np.bool),undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles)==n_tiles and 1<=n_tiles:
            largest_tiling_axis = [i for i in np.argsort(x.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i==largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles,undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)

        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
            _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        if any(np.array(n_tiles) != np.array(n_tiles_limited)):
            print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        n_tiles = n_tiles_limited


        # normalize & resize
        x = resizer.before(x, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(np.prod(n_tiles),1)
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                denoised, segmentation = self.predict_tiled(self.keras_model,x,axes_in=net_axes_in,axes_out=net_axes_out,
                                  n_tiles=n_tiles,block_sizes=net_axes_in_div_by,tile_overlaps=net_axes_in_overlaps,pbar=progress)
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(x.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                    raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles,undo=True)))
                progress.total = np.prod(n_tiles)

        denoised = resizer.after(denoised, net_axes_out)
        segmentation = resizer.after(segmentation, net_axes_out)

        return denoised, segmentation

    def _limit_tiling(self,img_shape,n_tiles,block_sizes):
        img_shape, n_tiles, block_sizes = np.array(img_shape), np.array(n_tiles), np.array(block_sizes)
        n_tiles_limit = np.ceil(img_shape / block_sizes) # each tile must be at least one block in size
        return [int(t) for t in np.minimum(n_tiles,n_tiles_limit)]

    def predict_tiled(self, keras_model, x, n_tiles, block_sizes, tile_overlaps, axes_in, axes_out=None, pbar=None, **kwargs):
        """TODO."""

        if all(t == 1 for t in n_tiles):
            denoised, segmentation = self.predict_direct(keras_model, x, axes_in, axes_out, **kwargs)
            if pbar is not None:
                pbar.update()
            return denoised, segmentation

        ###

        if axes_out is None:
            axes_out = axes_in
        axes_in, axes_out = axes_check_and_normalize(axes_in, x.ndim), axes_check_and_normalize(axes_out)
        assert 'S' not in axes_in
        assert 'C' in axes_in and 'C' in axes_out
        ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
        channel_in, channel_out = ax_in['C'], ax_out['C']

        assert set(axes_out).issubset(set(axes_in))
        axes_lost = set(axes_in).difference(set(axes_out))

        def _to_axes_out(seq, elem):
            # assumption: prediction size is same as input size along all axes, except for channel (and lost axes)
            assert len(seq) == len(axes_in)
            # 1. re-order 'seq' from axes_in to axes_out semantics
            seq = [seq[ax_in[a]] for a in axes_out]
            # 2. replace value at channel position with 'elem'
            seq[ax_out['C']] = elem
            return tuple(seq)

        ###

        assert x.ndim == len(n_tiles) == len(block_sizes)
        assert n_tiles[channel_in] == 1
        assert all(n_tiles[ax_in[a]] == 1 for a in axes_lost)
        assert all(np.isscalar(t) and 1 <= t and int(t) == t for t in n_tiles)

        # first axis > 1
        axis = next(i for i, t in enumerate(n_tiles) if t > 1)

        block_size = block_sizes[axis]
        tile_overlap = tile_overlaps[axis]
        n_block_overlap = int(np.ceil(1. * tile_overlap / block_size))

        # print(f"axis={axis},n_tiles={n_tiles[axis]},block_size={block_size},tile_overlap={tile_overlap},n_block_overlap={n_block_overlap}")

        n_tiles_remaining = list(n_tiles)
        n_tiles_remaining[axis] = 1

        def channel_axis_shuffle(_to_axes_out, channel_out, denoised, dst, s_dst, s_src, x):
            if dst is None:
                dst_shape = _to_axes_out(x.shape, denoised.shape[channel_out])
                dst = np.empty(dst_shape, dtype=x.dtype)
            s_src = _to_axes_out(s_src, slice(None))
            s_dst = _to_axes_out(s_dst, slice(None))
            dst[s_dst] = denoised[s_src]
            return dst

        dst_denoised = None
        dst_segmentation = None
        for tile, s_src, s_dst in tile_iterator_1d(x, axis=axis, n_tiles=n_tiles[axis], block_size=block_size,
                                                   n_block_overlap=n_block_overlap):

            denoised, segmentation = self.predict_tiled(keras_model, tile, n_tiles_remaining, block_sizes, tile_overlaps, axes_in, axes_out,
                                 pbar=pbar, **kwargs)

            denoised = channel_axis_shuffle(_to_axes_out, channel_out, denoised, dst_denoised, s_dst, s_src, x)
            segmentation = channel_axis_shuffle(_to_axes_out, channel_out, segmentation, dst_segmentation, s_dst, s_src, x)

        return denoised, segmentation



    def predict_direct(self, keras_model, x, axes_in, axes_out=None, **kwargs):
        """TODO."""
        if axes_out is None:
            axes_out = axes_in
        ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
        channel_in, channel_out = ax_in['C'], ax_out['C']
        single_sample = ax_in['S'] is None
        len(axes_in) == x.ndim or _raise(ValueError())
        x = to_tensor(x, channel=channel_in, single_sample=single_sample)
        denoised_tensor, segmentation_tensor = keras_model.predict(x, **kwargs)
        denoised = from_tensor(denoised_tensor, channel=channel_out, single_sample=single_sample)
        segmentation = from_tensor(segmentation_tensor, channel=channel_out, single_sample=single_sample)
        len(axes_out) == denoised.ndim or _raise(ValueError())
        return denoised, segmentation

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

        if self.config.train_loss == 'denoiseg':
            loss_n2v = denoiseg_denoise_loss()
            loss_seg = denoiseg_seg_loss(self.config.relative_weights)
        else:
            _raise('Unknown Loss!')

        callbacks = [TerminateOnNaN()]

        # compile model
        model.compile(optimizer=optimizer, loss={'out_denoise': loss_n2v, 'out_segment': loss_seg},
                      loss_weights={'out_denoise': self.config.denoiseg_alpha, 'out_segment': 1-self.config.denoiseg_alpha})

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

            
    @suppress_without_basedir(warn=True)
    def export_TF(self, name, description, authors, test_img, axes, patch_shape, fname=None):
        """
        name: String
            Name of the model. 
        description: String
            A short description of the model e.g. on what data it was trained.
        authors: String
            Comma seperated list of author names.
        patch_shape: The shape of the patches used in model.train().
        """
        if fname is None:
            fname = self.logdir / 'export.bioimage.io.zip'
        else:
            fname = Path(fname)
            
        input_n_dims = len(test_img.shape)
        if 'C' in axes:
            input_n_dims -=1 
        assert input_n_dims == self.config.n_dim, 'Input and network dimensions do not match.'
        assert test_img.shape[axes.index('X')] == test_img.shape[axes.index('Y')], 'X and Y dimensions are not of same length.'    
        test_output = self.predict(test_img, axes) 
        # Extract central slice of Z-Stack
        if 'Z' in axes:
            z_dim = axes.index('Z')
            if z_dim != 0:
                test_output = np.moveaxis(test_output, z_dim, 0)
            test_output = test_output[int(test_output.shape[0]/2)]
        
        # CSBDeep Export
        meta = {
            'type':          self.__class__.__name__,
            'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   self._axes_div_by(self.config.axes),
            'tile_overlap':  self._axes_tile_overlap(self.config.axes),
        }
        export_SavedModel(self.keras_model, str(fname), meta=meta)
        # CSBDeep Export Done
        
        # Replace : with -
        name = name.replace(':', ' -')
        yml_dict = self.get_yml_dict(name, description, authors, test_img, axes, patch_shape=patch_shape)
        yml_file = self.logdir / 'model.yaml'
        
        '''default_flow_style must be set to TRUE in order for the output to display arrays as [x,y,z]'''
        yaml = YAML(typ='rt') 
        yaml.default_flow_style = False
        with open(yml_file, 'w') as outfile:
            yaml.dump(yml_dict, outfile)
            
        input_file = self.logdir / 'testinput.tif'
        output_denoised_file = self.logdir / 'testoutput_denoise.tif'
        output_segmented_file = self.logdir / 'testoutput_segment.tif'
        imsave(input_file, test_img)
        imsave(output_denoised_file, test_output[0])
        imsave(output_segmented_file, test_output[1])
            
        with ZipFile(fname, 'a') as myzip:
            myzip.write(yml_file, arcname=os.path.basename(yml_file))
            myzip.write(input_file, arcname=os.path.basename(input_file))
            myzip.write(output_denoised_file, arcname=os.path.basename(output_denoised_file))
            myzip.write(output_segmented_file, arcname=os.path.basename(output_segmented_file))
            
        print("\nModel exported in BioImage ModelZoo format:\n%s" % str(fname.resolve()))
            
    
    def get_yml_dict(self, name, description, authors, test_img, axes, patch_shape=None):
        if (patch_shape != None):
            self.config.patch_shape = patch_shape
            
        ''' Repeated values to avoid reference tags of the form &id002 in yml output when the same variable is used more than
        once in the dictionary''' 
        mean_val = [] 
        mean_val1 = [] 
        for ele in self.config.means:
            mean_val.append(float(ele))
            mean_val1.append(float(ele))
        std_val = [] 
        std_val1 = [] 
        for ele in self.config.stds:
            std_val.append(float(ele))
            std_val1.append(float(ele))
        in_data_range_val = ['-inf', 'inf']
        out_data_range_val = ['-inf', 'inf']
            
        axes_val = 'b' + self.config.axes
        axes_val = axes_val.lower()
        val = 2**self.config.unet_n_depth
        val1 = predict.tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
        min_val = [1, val, val, self.config.n_channel_in ]
        step_val = [1, val, val, 0]
        halo_val = [0, val1, val1, 0]
        scale_val = [1, 1, 1, 1]
        offset_val = [0, 0, 0, 3]
        
        yaml = YAML(typ='rt')
        with open(self.logdir/'config.json','r') as f:
            tr_kwargs_val = yaml.load(f)
        
        if (self.config.n_dim == 3):
            min_val = [1, val, val, val, self.config.n_channel_in ]
            step_val = [1, val, val, val, 0]
            halo_val = [0, val1, val1, val1, 0]
            scale_val = [1, 1, 1, 1, 1]
            offset_val = [0, 0, 0, 0, 0]
 
        yml_dict = {
            'name': name,
            'description': description,
            'cite': [{
                'text': 'Tim-Oliver Buchholz and Mangal Prakash and Alexander Krull and Florian Jug DenoiSeg: Joint Denoising and Segmentation\nArXiv (2020)',
                'doi': 'arXiv:2005.02987'
            }],
            'authors': authors,
            'language': 'python',
            'framework': 'tensorflow',
            'format_version': '0.2.0-csbdeep',
            'source': 'denoiseg',
            'test_input': 'testinput.tif',
            'test_output': 'testoutput.tif',
            'inputs': [{
                'name': 'input',
                'axes': axes_val,
                'data_type': 'float32',
                'data_range': in_data_range_val,
                'halo': halo_val,
                'shape': {
                    'min': min_val,
                    'step': step_val
                }
            }],
            'outputs': [
                {
                    'name': self.keras_model.layers[-2].output.name,
                    'axes': axes_val,
                    'data_type': 'float32',
                    'data_range': out_data_range_val,
                    'shape': {
                        'reference_input': 'input',
                        'scale': scale_val,
                        'offset': offset_val
                    }
                },
                {
                    'name': self.keras_model.layers[-1].output.name,
                    'axes': axes_val,
                    'data_type': 'float32',
                    'data_range': out_data_range_val,
                    'shape': {
                        'reference_input': 'input',
                        'scale': scale_val,
                        'offset': offset_val
                    }
                }
            ],
            'training': {
                'source': 'n2v.train()',
                'kwargs': tr_kwargs_val
            },
            'prediction': {
                'weights': {'source': './variables/variables'},
                'preprocess': [{
                    'kwargs': { 
                        'mean': mean_val,
                        'stdDev': std_val
                    }
                }],
                'postprocess': [{
                    'kwargs': { 
                        'mean': mean_val1,
                        'stdDev': std_val1
                    }
                }]
            }
        }
        
        return yml_dict
    
    
    @property
    def _config_class(self):
        return DenoiSegConfig

    
def export_SavedModel(model, outpath, meta={}, format='zip'):
    """Export Keras model in TensorFlow's SavedModel_ format.
    See `Your Model in Fiji`_ to learn how to use the exported model with our CSBDeep Fiji plugins.
    .. _SavedModel: https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory
    .. _`Your Model in Fiji`: https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji
    Parameters
    ----------
    model : :class:`keras.models.Model`
        Keras model to be exported.
    outpath : str
        Path of the file/folder that the model will exported to.
    meta : dict, optional
        Metadata to be saved in an additional ``meta.json`` file.
    format : str, optional
        Can be 'dir' to export as a directory or 'zip' (default) to export as a ZIP file.
    Raises
    ------
    ValueError
        Illegal arguments.
    """

    def export_to_dir(dirname):
        if len(model.inputs) > 1 or len(model.outputs) > 1:
            warnings.warn('Found multiple input or output layers.')
        builder = tf.saved_model.builder.SavedModelBuilder(dirname)
        # use name 'input'/'output' if there's just a single input/output layer
        inputs  = dict(zip(model.input_names,model.inputs))   if len(model.inputs)  > 1 else dict(input=model.input)
        outputs = dict(zip(model.output_names,model.outputs)) if len(model.outputs) > 1 else dict(output=model.output)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)
        signature_def_map = { tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature }
        builder.add_meta_graph_and_variables(K.get_session(),
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
        builder.save()
        if meta is not None and len(meta) > 0:
            save_json(meta, os.path.join(dirname,'meta.json'))


    ## checks
    isinstance(model,keras.models.Model) or _raise(ValueError("'model' must be a Keras model."))
    # supported_formats = tuple(['dir']+[name for name,description in shutil.get_archive_formats()])
    supported_formats = 'dir','zip'
    format in supported_formats or _raise(ValueError("Unsupported format '%s', must be one of %s." % (format,str(supported_formats))))

    # remove '.zip' file name extension if necessary
    if format == 'zip' and outpath.endswith('.zip'):
        outpath = os.path.splitext(outpath)[0]

    if format == 'dir':
        export_to_dir(outpath)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpsubdir = os.path.join(tmpdir,'model')
            export_to_dir(tmpsubdir)
            shutil.make_archive(outpath, format, tmpsubdir, './')
