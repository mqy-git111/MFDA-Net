import os
import numpy as np
from scipy import ndimage
import GeodisTK
from skimage import io
import SimpleITK as sitk
from medpy import metric
import glob as gb
import pandas as pd
import cv2


def precision(predict, target):
    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fp = np.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        hd = metric.binary.hd(pred, gt)
        return hd
    else:
        gt = cv2.resize(gt, (512, 512))
        gt = (gt * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        hd = radius
        return hd

if __name__ == '__main__':
    maskpath = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/NII_TO_IMG/test/NPY/test_maskV1_big.npy'
    predpath = './result/fuse_way/merge_input/merge_input_big.npy'
    masks = np.load(maskpath)  # test的npy文件位置
    segs = np.load(predpath)
    HD = []
    JC = []
    PPV = []
    for t in range(masks.shape[0]):
        print(t)
        mask = masks[t]
        mask = np.where(mask > 0.5, 1, 0)
        seg = segs[t]
        seg = np.where(seg > 0.5, 1, 0)

        hd = calculate_metric_percase(seg, mask)
        HD.append(hd)
        jc = metric.binary.jc(seg, mask)
        JC.append(jc)
        ppv = precision(seg, mask)
        PPV.append(ppv)


    hd_mean = np.nanmean(HD)
    print("HD:", hd_mean)
    HD95.append(hd_mean)
    data1 = pd.DataFrame(HD)
    data1.to_csv('./result/fuse_way/merge_input/merge_input_big_hd.csv')

    jc_mean = np.nanmean(JC)
    print("JC:", jc_mean)
    JC.append(jc_mean)
    data2 = pd.DataFrame(JC)
    data2.to_csv('./result/fuse_way/merge_input/merge_input_big_jc.csv')

    ppv_mean = np.nanmean(PPV)
    print("PPV:", ppv_mean)
    PPV.append(ppv_mean)
    data3 = pd.DataFrame(PPV)
    data3.to_csv('./result/fuse_way/merge_input/merge_input_big_ppv.csv')