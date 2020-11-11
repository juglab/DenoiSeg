from csbdeep.internals.blocks import unet_block
import tensorflow as tf
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last
from keras import Input
from keras.layers import Conv3D, Conv2D, Add, Activation, Lambda
from keras.models import Model


def denoiseg_model(input_shape,
                   last_activation,
                   n_depth=2,
                   n_filter_base=16,
                   kernel_size=(3, 3, 3),
                   n_conv_per_depth=2,
                   activation="relu",
                   batch_norm=False,
                   dropout=0.0,
                   pool_size=(2, 2, 2),
                   n_channel_out=1,
                   residual=False):
    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim == 2 else Conv3D

    input = Input(input_shape, name="input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
    if residual:
        if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    denoised = Lambda(lambda t: t[...,:1], name='out_denoise')(final)
    segmentation = Lambda(lambda t: t[...,1:], name='out_segment')(final)

    return Model(inputs=input, outputs=[denoised, segmentation])
