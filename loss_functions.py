import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import image
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance


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

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def cross_entropy_loss_RCF(y_true, y_pred):
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    # y_true = gradient_mag(y_true)
    # one = tf.ones_like(y_true)
    # zero = tf.zeros_like(y_true)
    # y_true = tf.where(y_true > 0.5, one, zero)
    y_true = tf.cast(y_true, tf.float32)
    lamda = 1.2
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    alpha = lamda * count_pos / (count_neg + count_pos)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / alpha

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def deta_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    error = K.abs(y_pred_f - y_true_f)
    err_mean = K.mean(error)

    return err_mean

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) + distance(posmask)

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)

def suface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)

    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)
    denominator = y_true + y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1-generalized_dice_coeff(y_true, y_pred)

