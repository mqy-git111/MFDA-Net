from keras.layers import Activation, Reshape, Lambda, dot, add, Permute
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
import tensorflow as tf
import numpy as np

def a_ISSA_attention(a_layer, v_layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = v_layer.get_shape().as_list()[1]
    else:
        in_channel = v_layer.get_shape().as_list()[3]

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    a_layer = InterlacedSparseSelfAttention(a_layer)
    concate = my_concat([a_layer, v_layer])
    concate = Conv2D(in_channel, [1, 1], strides=[1, 1], data_format=data_format)(concate)

    return concate

def normal_reshape(x, shape):
    return  K.reshape(x, shape)

def normal_permute(x, pattern):
    return K.permute_dimensions(x, pattern)

def InterlacedSparseSelfAttention(x):

    P_h = 64
    P_w = 64
    # N, C, H, W = x.shape
    N = x.get_shape().as_list()[0]
    H = x.get_shape().as_list()[1]
    W = x.get_shape().as_list()[2]
    C = x.get_shape().as_list()[3]
    Q_h, Q_w = H // P_h, W // P_w # Q_h = 128, Q_w = 128

    #(N, 512, 512, 1)
    x = Lambda(normal_reshape, arguments={'shape':(-1, Q_h, P_h, Q_w, P_w, C)})(x)  #(N,64,8,64,8,1 )

    #Long-range Attention
    x = Lambda(normal_permute, arguments={'pattern':(0, 2, 4, 1, 3, 5)})(x)
    x = Lambda(normal_reshape, arguments={'shape':(-1, Q_h, Q_w, C)})(x)
    x = non_local_block(x)
    x = Lambda(normal_reshape, arguments={'shape': (-1, P_h, P_w, Q_h, Q_w, C)})(x)    #(N, 8,8,64,64,1)


    #Short-range Attention
    x = Lambda(normal_permute, arguments={'pattern':(0, 3, 4, 1, 2, 5)})(x)    #(N,64,64, 8,8,1)
    x = Lambda(normal_reshape, arguments={'shape':(-1, P_h, P_w, C)})(x)    #(N*64*64,8,8,1)
    x = non_local_block(x)
    x = Lambda(normal_reshape, arguments={'shape':(-1, Q_h, Q_w, P_h, P_w, C)})(x) # (N,64,64, 8,8,1)
    x = Lambda(normal_permute, arguments={'pattern':(0, 1, 3, 2, 4, 5)})(x)   #(N,64,8,64,8,1)
    x = Lambda(normal_reshape, arguments={'shape':(-1, H, W, C)})(x)  #(N,512,512,1)

    return x

def non_local_block(ip, intermediate_dim=None, compression=2, mode='embedded', add_residual=True):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


# if __name__ =="__main__":
#     x = np.load('./NII_TO_IMG/test/NPY/test_imgV1.npy')
#     x = x[85:86]
#     x = InterlacedSparseSelfAttention(x, 128, 128)
#     print(x.shape)