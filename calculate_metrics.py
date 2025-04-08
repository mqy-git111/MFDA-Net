import cv2
import os
import numpy as np
import pandas as pd
from skimage import io
from medpy import metric
import glob as gb
# import matplotlib.pyplot as plt


def dice_coef1(y_true, y_pred):
    smooth = 0.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


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
    # dice = dice_coef1(pred, gt)
    if pred.sum() > 0:
        hd = metric.binary.hd(pred, gt)
        return hd
    else:
        gt = cv2.resize(gt, (512, 512))
        gt = (gt * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        hd = radius * 0.95
        return hd

def calculate_metric_ASD(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        asd = metric.binary.asd(pred, gt)
        return asd
    else:
        return 0


def Sen_spe(maskpath, predpath):

    spacing = pd.read_csv("./NII_TO_IMG/test/NPY1/spacing_of_testimg.csv")
    spacing = spacing.values[:, 1]
    # spacing = spacing.tolist()

    sens = []
    spes = []

    result_os = []
    result_us = []

    Dice = []

    HD = []
    ASD = []
    JC = []
    PPV = []

    # mask_list = os.listdir(maskpath)
    # mask_list.sort(key=lambda x:int(x.split('.')[0]))

    masks = np.load(maskpath)  # test的npy文件位置
    preds = np.load(predpath)

    for t in range(masks.shape[0]):
        # print(filename)
        print(t)
        # mask = io.imread(maskpath + filename)
        # pred = io.imread(predpath + filename)

        # mask = mask/255
        # pred = pred/255

        mask = masks[t]
        mask = np.where(mask > 0.5, 1, 0)
        pred = preds[t]
        pred = np.where(pred > 0.5, 1, 0)

        hd = calculate_metric_percase(pred, mask)
        asd = calculate_metric_ASD(pred, mask)
        HD.append(hd)
        ASD.append(asd)
        jc = metric.binary.jc(pred, mask)
        JC.append(jc)
        # ppv = precision(pred, mask)
        # PPV.append(ppv)
        #
        mask_f = mask.flatten()
        pred_f = pred.flatten()

        # 计算灵敏度Sen
        tp = np.sum(mask_f * pred_f)
        sen = tp / np.sum(mask_f)

        # 计算特异度Spe
        mask_n = np.abs(mask_f - 1)
        seg_n = np.abs(pred_f - 1)
        tn = np.sum(mask_n * seg_n)
        spe = tn / np.sum(mask_n)

        # # 计算精确率Precision
        # pre = tp / np.sum(seg_f)

        os_1 = pred_f - (mask_f * pred_f)
        osum_1 = mask_f + pred_f
        over_rate = np.sum(os_1) / np.sum(osum_1 > 0)

        us_1 = mask_f - (mask_f * pred_f)
        under_rate = np.sum(us_1) / np.sum(osum_1 > 0)

        dice = dice_coef1(mask, pred)
        Dice.append(dice)

        sens.append(sen)
        spes.append(spe)

        result_os.append(over_rate)
        result_us.append(under_rate)

    sen_mean = np.nanmean(sens)
    sen_std = np.std(sens)
    spe_mean = np.nanmean(spes)
    spe_std = np.std(spes)
    print("Sen:",sen_mean)
    print("Sen_std:", sen_std)
    print("Spe:", spe_mean)
    print("Spe_std:", spe_std)

    # 把均值也写入表格文件
    sens.append(sen_mean)
    sens.append(sen_std)
    spes.append(spe_mean)
    spes.append(spe_std)
    #
    os_mean = np.nanmean(result_os)
    os_std = np.std(result_os)
    us_mean = np.nanmean(result_us)
    us_std = np.std(result_us)
    print("OR:", os_mean)
    print("OR_std:", os_std)
    print("UR:", us_mean)
    print("UR_std:", us_std)

    #把均值也写入表格文件
    result_os.append(os_mean)
    result_os.append(os_std)
    result_us.append(us_mean)
    result_us.append(us_std)

    aver = np.nanmean(Dice)
    dice_std = np.std(Dice)
    print("dice nums:", len(Dice))
    print("dice:", aver)
    print("dice_std:", dice_std)
    Dice.append(aver)
    Dice.append(dice_std)

    # HD = np.multiply(np.array(HD), spacing)
    # HD = HD.tolist()
    hd_mean = np.nanmean(HD)
    print("HD:", hd_mean)
    hd_std = np.std(HD)
    print("hd_std:", hd_std)
    HD.append(hd_mean)
    HD.append(hd_std)

    # ASD = np.multiply(np.array(ASD), spacing)
    # ASD = ASD.tolist()
    asd_mean = np.nanmean(ASD)
    asd_std = np.std(ASD)
    print("ASD:", asd_mean)
    print("asd_std:", asd_std)
    ASD.append(asd_mean)
    ASD.append(asd_std)

    jc_mean = np.nanmean(JC)
    jc_std = np.std(JC)
    print("JC:", jc_mean)
    print("JC_std:", jc_std)
    JC.append(jc_mean)
    JC.append(jc_std)

    # ppv_mean = np.nanmean(PPV)
    # print("PPV:", ppv_mean)
    # PPV.append(ppv_mean)

    return sens, spes, result_os, result_us, Dice, HD, ASD, JC, PPV


if __name__ == '__main__':
    mask_floder = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/NII_TO_IMG/test/NPY/test_maskV1_big.npy'
    pred_floder = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/result/jiance_a_v_ronghe/block_2/a_concate_v_big.npy'
    sen, spe, OR, UR, Dice, hd, asd, jc, ppv =Sen_spe(mask_floder, pred_floder)
    # Sen = pd.DataFrame(sen)
    # Spe = pd.DataFrame(spe)
    # Sen.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_sensitivity.csv')
    # Spe.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_specificity.csv')
    # os_rate = pd.DataFrame(OR)
    # us_rate = pd.DataFrame(UR)
    # os_rate.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_overrate.csv')
    # us_rate.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_underrate.csv')
    # data1 = pd.DataFrame(Dice)
    # data1.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_dice.csv')

    data1 = pd.DataFrame(hd)
    data1.to_csv('/media/lao/C14D581BDA18EBFA1/xss/work2/result/unet_mff/unet_mff_hd.csv')
    # data1 = pd.DataFrame(asd)
    # data1.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_asd.csv')
    # data2 = pd.DataFrame(jc)
    # data2.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_jc.csv')
    # data3 = pd.DataFrame(ppv)
    # data3.to_csv('/media/lao/C14D581BDA18EBFA1/xss/Unet/work1/result/MIX/mix_p/mix_p_ppv.csv')