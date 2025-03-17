import cv2
import os
import numpy as np
from skimage import io
import glob as gb
import matplotlib.pyplot as plt



def edge_extract(imgpath, maskpath, resultpath,resultpath1, edgepath):

    pathNames = gb.glob(imgpath + '*.png')
    for path_name in pathNames:

        # name = path_name[path_name.rfind("/"):-4]
        name = str(3804)
        print(name)
        img = io.imread(imgpath + name + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = io.imread(maskpath + name + '.png')
        # seg = io.imread(resultpath + name + '.png')
        # seg1 = io.imread(resultpath1 + name + '.bmp')
        # kernel = np.ones((5, 5), np.uint8)
        # seg_1 = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)

        contours_mask, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[-1]
        cv2.drawContours(img, contours_mask, -1, (0, 255, 0), 2)  #mask
        # contours_seg, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours_seg, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)   #seg红色
        # contours_seg_1, hierarchy = cv2.findContours(seg1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours_seg_1, -1, (51, 153, 255), 2, lineType=cv2.LINE_AA)
        cv2.imwrite(edgepath + name + '.png', img)

    return 0


if __name__ == '__main__':
    # imgpath = './liver/test/nid_image/'
    # maskpath = './liver/test/nid_mask/'
    # resultpath = './liver/test/liver_nid_result/'
    # edgepath = './liver/test/liver_nid_edge'

    imgpath = '/media/lao/C14D581BDA18EBFA1/xss/Unet/work3/data/Tumor/train/images_v/'
    maskpath = '/media/lao/C14D581BDA18EBFA1/xss/Unet/work3/data/Tumor/train/labels/'   #绿色
    resultpath = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/result/da_add_av/block_2/pred_big/'   #红色
    resultpath1 = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/result/baseline/small/pred/'   #蓝色
    edgepath = '/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/result/da_add_av/block_2/gam/edge/'
    edge_extract(imgpath, maskpath, resultpath,resultpath1, edgepath)