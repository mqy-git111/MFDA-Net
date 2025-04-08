import numpy as np
import nibabel as nib
import copy
import cv2
import scipy
import scipy.ndimage
from scipy.ndimage import rotate
from skimage.transform import resize
from scipy.ndimage import measurements
import random
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom as sni
import SimpleITK as sitk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import image


# 缩放
def scaleit(image, factor, isseg=False):
    order = 0 if isseg is True else 3
    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth


    if factor < 1.0:
        newimg = np.zeros_like(image)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        new = sni(image, [float(factor), float(factor), 1],
                               order=order, mode='nearest')
        new = new[0:zheight, 0:zwidth, 0:zdepth]
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = new
        return newimg

    elif factor > 1.0:
        newimg = sni(image, [float(factor), float(factor), 1], order=order, mode='nearest')
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        return newimg
    else:
        return image


# 旋转
def rotateit(image, angle, isseg=False):
    order = 0 if isseg is True else 3
    return sni.rotate(image, float(angle), reshape=False, order=order, mode='nearest')


# 平移
def shiftit(image, length, isseg=False):
    order = 0 if isseg is True else 3
    return sni.shift(image, [float(length), float(length), 0], output=None, order=order, mode='constant', cval=0.0,
          prefilter=True)

# 对比度
def augment_contrast(data_sample, r, contrast_range=(0.75, 1.25), preserve_range=True):
    mn = data_sample.mean()
    if preserve_range:
        minm = data_sample.min()
        maxm = data_sample.max()
    if r < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
    # factor = contrast_range[1]
    data_sample = (data_sample - mn) * factor + mn
    if preserve_range:
        data_sample[data_sample < minm] = minm
        data_sample[data_sample > maxm] = maxm
    return data_sample


# 亮度
def augment_brightness(data_sample, r):
    if r < 0.5:
        rnd_nb = np.random.normal(0, 0.1)
        data_sample += rnd_nb
    else:
        multiplier_range = (0.5, 2)
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        data_sample *= multiplier
    # multiplier = 1.5
    # data_sample *= multiplier

    return data_sample


# gamma
def augment_gamma(data_sample, r,  gamma_range=(0.5, 2), epsilon=1e-7, retain_stats=False):

    if retain_stats:
        mn = data_sample.mean()
        sd = data_sample.std()
    if r < 0.5 and gamma_range[0] < 1:
        gamma = np.random.uniform(gamma_range[0], 1)
    else:
        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
    minm = data_sample.min()
    rnge = data_sample.max() - minm
    data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
    if retain_stats:
        data_sample = data_sample - data_sample.mean() + mn
        data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    return data_sample

# 噪声
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):

    # If size is None, then a single value is generated and returned.
    if random_state is None:
        random_state = np.random.RandomState(None)

    # The form of shape is as follows (weight, heght, channels)
    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    # 对于仿射变换，我们只需要知道变换前的三个点与其对应的变换后的点，就可以通过cv2.getAffineTransform求得变换矩阵.
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    # pts1 是变换前的三个点，pts2 是变换后的三个点
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    # 进行放射变换
    M = cv2.getAffineTransform(pts1, pts2)
    # print(image.shape)
    # 默认使用 双线性插值，这里使用 三次样条插值。处理速度会变慢，但是可以最大程度的保留细节，适用于医学图像
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_CUBIC)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=3, mode='reflect').reshape(shape)

###########################
def data_aug(img_norm_v, label_temp, aug_type):
    if aug_type == 0:
        rand_angle = [-30, 30]
        np.random.shuffle(rand_angle)
        img_norm_v = rotate(img_norm_v, angle=rand_angle[0], axes=(1, 0), reshape=False)
        # img_norm_a = rotate(img_norm_a, angle=rand_angle[0], axes=(1, 0), reshape=False)
        label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False)
    if aug_type == 1:
        rand_angle = [-15, 15]
        np.random.shuffle(rand_angle)
        img_norm_v = rotate(img_norm_v, angle=rand_angle[0], axes=(1, 0), reshape=False)
        # img_norm_a = rotate(img_norm_a, angle=rand_angle[0], axes=(1, 0), reshape=False)
        label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False)

    # scale
    elif aug_type == 2:
        factor = np.random.random() * 0.6 + 0.8
        img_norm_v = scaleit(img_norm_v, factor)
        # img_norm_a = scaleit(img_norm_a, factor)
        label_temp = scaleit(label_temp, factor, True)
    # contrast
    elif aug_type == 3:
        r = np.random.random()
        img_norm_v = augment_contrast(img_norm_v, r)
        # img_norm_a = augment_contrast(img_norm_a, r)
    # bright
    elif aug_type == 4:
        r = np.random.random()
        img_norm_v = augment_brightness(img_norm_v, r)
        # img_norm_a = augment_brightness(img_norm_a, r)
    # gamma
    elif aug_type == 5:
        r = np.random.random()
        img_norm_v = augment_gamma(img_norm_v, r)
        # img_norm_a = augment_gamma(img_norm_a, r)
    # noise
    elif aug_type == 6:
        img_norm_v = augment_gaussian_noise(img_norm_v)
        # img_norm_a = augment_gaussian_noise(img_norm_a)
    elif aug_type == 7:
        img_norm_v = img_norm_v
        # img_norm_a = img_norm_a
        label_temp = label_temp
    elif aug_type == 8:
        im_merge = np.concatenate((img_norm_v, label_temp), axis=2)
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
        im_t_v = im_merge_t[:, :, 0]
        # im_t_a = im_merge_t[:, :, 1]
        im_mask_t = im_merge_t[:, :, 1]
        img_norm_v = im_t_v.reshape((im_t_v.shape[0], im_t_v.shape[1], 1))
        # img_norm_a = im_t_a.reshape((im_t_a.shape[0], im_t_a.shape[1], 1))
        label_temp = im_mask_t.reshape((im_mask_t.shape[0], im_mask_t.shape[1], 1))
    return img_norm_v, label_temp

def v_data_aug(img_norm_v, label_temp, aug_type):
    if aug_type == 0:
        rand_angle = [-30, 30]
        np.random.shuffle(rand_angle)
        img_norm_v = rotate(img_norm_v, angle=rand_angle[0], axes=(1, 0), reshape=False)
        label_temp = rotate(label_temp, angle=rand_angle[0], axes=(1, 0), reshape=False)
    # scale
    elif aug_type == 1:
        factor = np.random.random() * 0.6 + 0.8
        img_norm_v = scaleit(img_norm_v, factor)
        label_temp = scaleit(label_temp, factor, True)
    # contrast
    elif aug_type == 2:
        r = np.random.random()
        img_norm_v = augment_contrast(img_norm_v, r)
    # bright
    elif aug_type == 3:
        r = np.random.random()
        img_norm_v = augment_brightness(img_norm_v, r)
    # gamma
    elif aug_type == 4:
        r = np.random.random()
        img_norm_v = augment_gamma(img_norm_v, r)
    # noise
    elif aug_type == 5:
        img_norm_v = augment_gaussian_noise(img_norm_v)
    elif aug_type == 6:
        img_norm_v = img_norm_v
        label_temp = label_temp
    elif aug_type == 7:
        im_merge = np.concatenate((img_norm_v, label_temp), axis=2)
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
        im_t = im_merge_t[:, :, 0]
        im_mask_t = im_merge_t[:, :, 1]
        img_norm = im_t.reshape((im_t.shape[0], im_t.shape[1], 1))
        label_temp = im_mask_t.reshape((im_mask_t.shape[0], im_mask_t.shape[1], 1))
    return img_norm_v, label_temp