from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

ce_w = 0.8
ce_d_w = 0.3
e = K.epsilon()
smooth = 1.0
'''
ce_w values smaller than 0.5 penalize false positives more while values larger than 0.5 penalize false negatives more
ce_d_w is level of contribution of the cross-entropy loss in the total loss.
'''

def Combo_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    d = 1.0 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
    out = - ((ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f)))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) + ((1 - ce_d_w) * d)
    return combo


def Dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def Dice_Loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss = 1.0 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return dice_loss

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

def Hausdorff_distance_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    y_pred_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_pred],
                                     Tout=tf.float32)
    pred_error = K.square(y_true - y_pred)
    distance = y_pred_dist_map ** 2 + y_true_dist_map ** 2
    dt_field = pred_error * distance
    hd_loss = K.mean(dt_field)
    return hd_loss

def dice_hd_loss(y_true, y_pred):
    dice_loss = Dice_Loss(y_true, y_pred)
    hd_loss = Hausdorff_distance_loss(y_true, y_pred)
    loss = dice_loss + hd_loss
    return loss




