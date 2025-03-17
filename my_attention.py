from keras.layers import Activation, Conv2D, Add, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import multiply, concatenate, add, subtract
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from keras.layers.core import Lambda


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, training=True):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, kernel_initializer='he_normal', trainable=True)(input)
        c = Conv2D(filters // 8, 1, kernel_initializer='he_normal', trainable=True)(input)
        d = Conv2D(filters, 1, kernel_initializer='he_normal', trainable=True)(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        # out = self.gamma * bcTd
        out = self.gamma*bcTd + input
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, training=True):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        # out = self.gamma * aaTa
        out = self.gamma*aaTa + input
        return out

def da_attention(x, name, train=True):
    in_channel = x.get_shape().as_list()[3]
    cam = CAM()(x)
    pam = PAM()(x)
    cam = Conv2D(in_channel, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name+'cam', trainable=train)(cam)
    pam = Conv2D(in_channel, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name+'pam', trainable=train)(pam)
    out = Add(name=name+'add')([cam, pam])
    out = Conv2D(in_channel, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name+'out', trainable=train)(out)
    da_out = Activation('relu')(out)
    return da_out

def attention_up_and_concate(v_layer, a_layer, name, train = True, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = v_layer.get_shape().as_list()[1]
    else:
        in_channel = v_layer.get_shape().as_list()[3]

    att_a = attention_block_2d(x=v_layer, g=a_layer, inter_channel=in_channel // 4, name=name, train=train, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1), name=name+'concat')
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3), name=name+'concat')

    concate = my_concat([v_layer, att_a])

    concate = Conv2D(in_channel, [1, 1], strides=[1, 1], activation='relu', kernel_initializer='he_normal',
                     data_format=data_format, name=name+'av_fuse', trainable=train)(concate)

    return concate


def attention_block_2d(x, g, inter_channel, name, train, data_format='channels_first'):

    theta_x = Conv2D(inter_channel, [1, 1],
                     strides=[1, 1],
                     kernel_initializer='he_normal',
                     data_format=data_format,
                     name=name+'theta_v',
                     trainable=train)(x)

    phi_g = Conv2D(inter_channel,[1, 1],
                   strides=[1, 1],
                   kernel_initializer='he_normal',
                   data_format=data_format,
                   name=name+'phi_a',
                   trainable=train)(g)

    f = add([theta_x, phi_g], name=name+'add')

    f = Activation('relu', name=name+'relu')(f)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format, name=name+'psi_f', trainable=train)(f)

    rate = Activation('sigmoid', name=name+'rate')(psi_f)

    att_g = multiply([g, rate], name=name+'a_att')

    return att_g

def feature_4_add(f1, f2, f3, f4):

    path_1 = Conv2D(8, 2, strides=[2, 2], activation='relu', padding='same', kernel_initializer='he_normal')(f1)

    f2_1 = Conv2D(16, 1, kernel_initializer='he_normal')(f2)
    f2_1 = concatenate([path_1, f2_1], axis=3)
    path_2 = Conv2D(16, 2, strides=[2, 2], activation='relu', padding='same', kernel_initializer='he_normal')(f2_1)

    f3_1 = Conv2D(32, 1, kernel_initializer='he_normal')(f3)
    f3_1 = concatenate([path_2, f3_1], axis=3)
    path_3 = Conv2D(32, 2, strides=[2, 2], activation='relu', padding='same', kernel_initializer='he_normal')(f3_1)

    path_4 = Conv2D(64, 1, kernel_initializer='he_normal')(f4)
    att_path = concatenate([path_3, path_4], axis=3)
    att_path = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(att_path)

    return att_path

def feature_fuse(att_path, x):

    channel = x.get_shape().as_list()[3]
    scale = att_path.get_shape().as_list()[1] // x.get_shape().as_list()[1]
    f = Conv2D(64, 1, kernel_initializer='he_normal')(x)
    att_gate = att_path
    k = scale//2
    for i in range(k):
        att_gate = Conv2D(64, 2, strides=[2, 2], activation='relu', padding='same', kernel_initializer='he_normal')(att_gate)
    x_rate = Conv2D(64, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(att_gate)
    f_fuse = multiply([x_rate, f])
    x_return = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))([x, f_fuse])
    x_return = Conv2D(channel, 1, activation='relu', kernel_initializer='he_normal')(x_return)
    return x_return

def concate_fuse(att_path, encoder):
    scale = encoder.get_shape().as_list()[1] // att_path.get_shape().as_list()[1]
    channel = encoder.get_shape().as_list()[3]
    att = Conv2DTranspose(channel, scale, strides=scale, padding='same', activation='relu')(att_path)
    att_rate = Conv2D(channel, 1, activation='sigmoid', padding='same')(att)
    att_encoder = multiply([att_rate, encoder])
    att_encoder = add([att_encoder, encoder])
    return att_encoder

def sub_path(up, down):
    channel = up.get_shape().as_list()[3]
    down_de = Conv2D(channel, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(down))
    sub_feature = subtract([up, down_de])
    sub_feature = Conv2D(channel, 3, activation='relu', padding='same', kernel_initializer='he_normal')(sub_feature)
    return sub_feature


