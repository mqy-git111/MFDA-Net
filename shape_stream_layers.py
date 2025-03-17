from keras.layers import Activation, Conv2D, Add, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import multiply, concatenate, add, subtract
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from keras.layers.core import Lambda


def resize_to(x, target_t):
    channels = x.get_shape().as_list()[3]
    scale = target_t.get_shape().as_list()[1]//x.get_shape().as_list()[1]
    x = Conv2DTranspose(filters=channels,
                        kernel_size=scale,
                        strides=scale)(x)
    return x


def _all_close(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def gradient_mag(tensor, from_rgb=False, eps=1e-12):
    if from_rgb:
        tensor = tf.image.rgb_to_grayscale(tensor[..., :3])
    tensor_edge = tf.image.sobel_edges(tensor)

    def _normalised_mag():
        mag = tf.reduce_sum(tensor_edge ** 2, axis=-1) + eps
        mag = tf.math.sqrt(mag)
        mag /= tf.reduce_max(mag, axis=[1, 2], keepdims=True)
        return mag

    z = tf.zeros_like(tensor)
    normalised_mag = tf.cond(
        _all_close(tensor_edge, tf.zeros_like(tensor_edge)),
        lambda: z,
        _normalised_mag, name='potato')

    return normalised_mag

# def GateConv(x):
#     in_channels = x.get_shape().as_list()[-1]
#     x = Conv2D(filters=in_channels//2,
#                kernel_size=3,
#                activation='relu',
#                padding='same',
#                kernel_initializer='he_normal')(x)
#     x = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)
#     return x
#
# def GatedShapeConv(x1, x2):
#     feature_channels = x1.get_shape().as_list()[-1]
#     features = concatenate([x1, x2], axis=-1)
#     alpha = GateConv(features)
#     gated = x1 * (alpha + 1.)
#     gated = Conv2D(feature_channels, 1)(gated)
#     return gated
#
# def ResnetPreactUnit(x):
#     cs = x.get_shape().as_list()[-1]
#     shortcut = x
#     x = Conv2D(filters=cs, kernel_size=3, padding='same', activation='relu')(x)
#     x = Conv2D(filters=cs, kernel_size=3, padding='same', activation='relu')(x)
#     x = Add()([shortcut, x])
#     return x
#
# def ShapeAttention(x):
#     s1, s2, s3, s4, s5 = x
#     s1 = Conv2D(filters=16, kernel_size=1)(s1)
#     s2 = Conv2D(filters=16, kernel_size=1)(s2)
#     s3 = Conv2D(filters=16, kernel_size=1)(s3)
#     s4 = Conv2D(filters=16, kernel_size=1)(s4)
#     s5 = Conv2D(filters=16, kernel_size=1)(s5)
#
#     shortcut1 = s1
#     s1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(s1)
#     s1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(s1)
#     x = Add()([shortcut1, s1])
#     s2 = resize_to(s2, target_t=x)
#     # gated_conv_1
#     features1 = concatenate([x, s2], axis=-1)
#     alpha1 = Conv2D(filters=16,
#                kernel_size=3,
#                activation='relu',
#                padding='same',
#                kernel_initializer='he_normal')(features1)
#     alpha1 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(alpha1)
#     gated1 = x * (alpha1 + 1.)
#     gated1 = Conv2D(filters=16, kernel_size=1, name='gate_1')(gated1)
#
#     shortcut2 = gated1
#     s2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(gated1)
#     s2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(s2)
#     x = Add()([shortcut2, s2])
#     s3 = resize_to(s3, target_t=x)
#     # gated_conv_1
#     features2 = concatenate([x, s3], axis=-1)
#     alpha2 = Conv2D(filters=16,
#                     kernel_size=3,
#                     activation='relu',
#                     padding='same',
#                     kernel_initializer='he_normal')(features2)
#     alpha2 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(alpha2)
#     gated2 = x * (alpha2 + 1.)
#     gated2 = Conv2D(filters=16, kernel_size=1, name='gate_2')(gated2)
#
#     shortcut3 = gated2
#     s3 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(gated2)
#     s3 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(s3)
#     x = Add()([shortcut3, s3])
#     s4 = resize_to(s4, target_t=x)
#     # gated_conv_1
#     features3 = concatenate([x, s4], axis=-1)
#     alpha3 = Conv2D(filters=16,
#                     kernel_size=3,
#                     activation='relu',
#                     padding='same',
#                     kernel_initializer='he_normal')(features3)
#     alpha3 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(alpha3)
#     gated3 = x * (alpha3 + 1.)
#     gated3 = Conv2D(filters=16, kernel_size=1, name='gate_3')(gated3)
#
#     shortcut4 = gated3
#     s4 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(gated3)
#     s4 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(s4)
#     x = Add()([shortcut4, s4])
#     s5 = resize_to(s5, target_t=x)
#     features4 = concatenate([x, s5], axis=-1)
#     alpha4 = Conv2D(filters=16,
#                     kernel_size=3,
#                     activation='relu',
#                     padding='same',
#                     kernel_initializer='he_normal')(features4)
#     alpha4 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(alpha4)
#     gated4 = x * (alpha4 + 1.)
#     gated4 = Conv2D(filters=16, kernel_size=1, name='gate_4')(gated4)
#
#     return gated4


class GateConv(Layer):

    def __init__(self, **kwargs):
        super(GateConv, self).__init__(**kwargs)
        self.conv_1 = None
        self.conv_2 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_1 = Conv2D(filters=in_channels//2,
                             kernel_size=3,
                             activation='relu',
                             padding='same',
                             kernel_initializer='he_normal')

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def call(self, x, training=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class GatedShapeConv(Layer):

    def __init__(self, **kwargs):
        super(GatedShapeConv, self).__init__(**kwargs)
        self.conv_1 = None
        self.gated_conv = GateConv()

    def build(self, input_shape):
        feature_channels = input_shape[0][-1]
        self.conv_1 = Conv2D(feature_channels, 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x, training=True):
        feature_map, shape_map = x
        features = concatenate([feature_map, shape_map], axis=-1)
        alpha = self.gated_conv(features, training=training)
        gated = feature_map*(alpha + 1.)
        return self.conv_1(gated)



class ResnetPreactUnit(Layer):

    def __init__(self, **kwargs):
        super(ResnetPreactUnit, self).__init__(**kwargs)
        self.conv_1 = None
        self.conv_2 = None
        self.add = Add()

    def build(self, input_shape):
        cs = input_shape[-1]

        self.conv_1 = Conv2D(filters=cs, kernel_size=3, padding='same', activation='relu')
        self.conv_2 = Conv2D(filters=cs, kernel_size=3, padding='same', activation='relu')

    def call(self, x, training=True):
        shortcut = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.add([x, shortcut])

def ShapeAttention(x):
    gated_conv_1 = GatedShapeConv()
    gated_conv_2 = GatedShapeConv()
    gated_conv_3 = GatedShapeConv()
    gated_conv_4 = GatedShapeConv()

    res_1 = ResnetPreactUnit()
    res_2 = ResnetPreactUnit()
    res_3 = ResnetPreactUnit()
    res_4 = ResnetPreactUnit()

    s1, s2, s3, s4, s5 = x
    s1 = Conv2D(filters=16, kernel_size=1)(s1)
    s2 = Conv2D(filters=16, kernel_size=1)(s2)
    s3 = Conv2D(filters=16, kernel_size=1)(s3)
    s4 = Conv2D(filters=16, kernel_size=1)(s4)
    s5 = Conv2D(filters=16, kernel_size=1)(s5)

    # todo these blocks should be a layer
    x = res_1(s1)
    # x = Conv2D(filters=16, kernel_size=1)(x)
    s2 = resize_to(s2, target_t=x)
    x = gated_conv_1([x, s2])

    x = res_2(x)
    # x = Conv2D(filters=16, kernel_size=1)(x)
    s3 = resize_to(s3, target_t=x)
    x = gated_conv_2([x, s3])

    x = res_3(x)
    # x = Conv2D(filters=16, kernel_size=1)(x)
    s4 = resize_to(s4, target_t=x)
    x = gated_conv_3([x, s4])

    x = res_4(x)
    # x = Conv2D(filters=16, kernel_size=1)(x)
    s5 = resize_to(s5, target_t=x)
    x = gated_conv_4([x, s5])


    # x = Conv2D(filters=1, kernel_size=1, name='edge_out')(x)

    return x
# class ShapeAttention(Layer):
#
#     def __init__(self, **kwargs):
#         super(ShapeAttention, self).__init__(**kwargs)
#
#         self.gated_conv_1 = GatedShapeConv()
#         self.gated_conv_2 = GatedShapeConv()
#         self.gated_conv_3 = GatedShapeConv()
#
#         self.shape_reduction_2 = Conv2D(
#             filters=1,
#             kernel_size=1,
#             )
#         self.shape_reduction_3 = Conv2D(
#             filters=1,
#             kernel_size=1,
#             )
#         self.shape_reduction_4 = Conv2D(
#             filters=1,
#             kernel_size=1,
#             )
#
#         self.res_1 = ResnetPreactUnit()
#         self.res_2 = ResnetPreactUnit()
#         self.res_3 = ResnetPreactUnit()
#
#         self.reduction_conv_1 = Conv2D(
#             filters=32,
#             kernel_size=1
#             )
#         self.reduction_conv_2 = Conv2D(
#             filters=16,
#             kernel_size=1
#             )
#         self.reduction_conv_3 = Conv2D(
#             filters=8,
#             kernel_size=1
#             )
#         self.reduction_conv_4 = Conv2D(
#             filters=1,
#             kernel_size=1,
#             activation='sigmoid'
#             )
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0][:-1] + (1,)
#
#     def call(self, x, training=None):
#         s1, s2, s3, s4 = x
#         s2 = self.shape_reduction_2(s2)
#         s3 = self.shape_reduction_3(s3)
#         s4 = self.shape_reduction_4(s4)
#
#         # todo these blocks should be a layer
#         x = self.res_1(s1, training=training)
#         x = self.reduction_conv_1(x)
#         s2 = resize_to(s2, target_t=x)
#         x = self.gated_conv_1([x, s2], training=training)
#
#         x = self.res_2(x, training=training)
#         x = self.reduction_conv_2(x)
#         s3 = resize_to(s3, target_t=x)
#         x = self.gated_conv_2([x, s3], training=training)
#
#         x = self.res_3(x, training=training)
#         x = self.reduction_conv_3(x)
#         s4 = resize_to(s4, target_t=x)
#         x = self.gated_conv_3([x, s4], training=training)
#
#         x = self.reduction_conv_4(x)
#
#         return x
#

# class ShapeStream(Layer):
#
#     def __init__(self, **kwargs):
#         super(ShapeStream, self).__init__(**kwargs)
#         self.shape_attention = ShapeAttention()
#         self.reduction_conv = Conv2D(
#             filters=1,
#             kernel_size=1,
#             activation='sigmoid'
#             )
#
#     def compute_output_shape(self, input_shape):
#         shape_intermediate_feats, _ = input_shape
#         return shape_intermediate_feats[0][:-1] + (1,)
#
#     def call(self, x, training=None):
#         shape_backbone_activations, image_edges = x
#         edge_out = self.shape_attention(shape_backbone_activations, training=training)
#         image_edges = resize_to(image_edges, target_t=edge_out)
#         backbone_representation = concatenate([edge_out, image_edges], axis=-1)
#         shape_attention = self.reduction_conv(backbone_representation)
#         return shape_attention, edge_out
def shape_stream(f1, f2, f3, f4, f5):
    edge_feature = ShapeAttention([f1, f2, f3, f4, f5])
    # image_edges = resize_to(img_edge, target_t=edge_out)
    # backbone_representation = concatenate([edge_out, image_edges], axis=-1)
    # shape_attention = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(backbone_representation)
    return edge_feature

def mask_edge_fuse(edge, mask):
    mask_pred = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal')(mask)
    edge_merge_mask = concatenate([mask_pred, edge], axis=-1)
    fuse_conv = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(edge_merge_mask)
    fuse_conv = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(fuse_conv)
    return fuse_conv