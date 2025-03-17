import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import nibabel as nib
import cv2
from A_V_unet import myUnet
import imageio
from PIL import Image
import pandas as pd



def nii_to_image(filepath, filepath1, maskpath, savepath):

    patpath = os.listdir(filepath)    #patpath-->BAIPUJIN....
    # patpath1 = os.listdir(filepath1)
    patmask = os.listdir(maskpath)    #patmask-->BAIPUJIN....

    img_npy = []
    # img_npy1 = []
    mask_npy = []
    center = 50  # 肺部的窗宽窗位
    width = 300

    pixel = []
    for patname in patpath:  #病人文件夹
        niipath = filepath +'/'+ patname
        niipath1 = filepath1 + '/' + patname
        patmask = maskpath +'/'+ patname
        niinames = os.listdir(niipath)    #niipath-->20180502A.nii.gz...
        niinames1 = os.listdir(niipath1)
        masknames = os.listdir(patmask)   #patmask-->20180502VS.nii.gz...把对应文件名改成一致
        for f in niinames:
            mask_path = os.path.join(patmask, f)
            img_path = os.path.join(niipath, f)
            img_path1 = os.path.join(niipath1, f)

            mask = nib.load(mask_path)   #读取nii
            img = nib.load(img_path)
            img1 = nib.load(img_path1)

            mask_fdata = mask.get_fdata()
            img_fdata = img.get_fdata()   #读出的即为CT值
            img_fdata1 = img1.get_fdata()

            spacing = -img_fdata.affine[0][0]

            # 转换成窗宽窗位
            min = (2 * center - width) / 2.0 + 0.5
            max = (2 * center + width) / 2.0 + 0.5
            dFactor = 1.0 / (max - min)

            img_fdata[img_fdata < min] = min
            img_fdata[img_fdata > max] = max
            img_fdata1[img_fdata1 < min] = min
            img_fdata1[img_fdata1 > max] = max

            # 进行转置，因为需要按照原来的方向进行保存
            # data = np.transpose(img_fdata, [2, 1, 0])
            (x, y, z, t) = img_fdata.shape


            for i in range(z):  # z是图像的序列
                mask_slice = mask_fdata[:,:,i]
                if np.max(mask_slice) > 0 and np.sum(mask_slice==1) > 5:
                    mask_slice = np.rot90(mask_slice, -1)
                    mask_slice = np.fliplr(mask_slice)
                    mask_slice = mask_slice.reshape((mask_slice.shape[0], mask_slice.shape[1], 1))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    mask_slice = cv2.morphologyEx(mask_slice, cv2.MORPH_CLOSE, kernel)
                    # mask_slice = cv2.resize(mask_slice, (256, 256))
                    mask_slice = mask_slice.reshape((mask_slice.shape[0], mask_slice.shape[1], 1))
                    mask_slice = mask_slice.astype('float32')
                    mask_npy.append(mask_slice)
                    p = np.sum(mask_slice==1)
                    pixel.append(p)

                    img_slice = img_fdata[:, :, i]
                    img_slice = img_slice - min
                    img_slice = img_slice * dFactor
                    img_slice[img_slice < 0.0] = 0
                    img_slice[img_slice > 1.0] = 1  # 转换为窗位窗位之后的数据
                    img_slice = np.rot90(img_slice, -1)
                    img_slice = np.fliplr(img_slice)
                    img_slice = img_slice.reshape((img_slice.shape[0], img_slice.shape[1], 1))
                    # img_slice = cv2.resize(img_slice, (256, 256))
                    # img_slice = img_slice.reshape((img_slice.shape[0], img_slice.shape[1], 1))
                    img_slice = img_slice.astype('float32')

                    img_slice1 = img_fdata1[:, :, i]
                    img_slice1 = img_slice1 - min
                    img_slice1 = img_slice1 * dFactor
                    img_slice1[img_slice1 < 0.0] = 0
                    img_slice1[img_slice1 > 1.0] = 1  # 转换为窗位窗位之后的数据
                    img_slice1 = np.rot90(img_slice1, -1)
                    img_slice1 = np.fliplr(img_slice1)
                    img_slice1 = img_slice1.reshape((img_slice1.shape[0], img_slice1.shape[1], 1))
                    # img_slice1 = cv2.resize(img_slice1, (256, 256))
                    # img_slice1 = img_slice1.reshape((img_slice1.shape[0], img_slice1.shape[1], 1))
                    img_slice1 = img_slice1.astype('float32')

                    a = myUnet()
                    model = a.get_unet()
                    model.load_weights('/media/lao/C14D581BDA18EBFA1/xss/Unet/XU_U_NET_pat/result/da_add_av/block_2/da_add_av.h5')
                    imgs_test_big = model.predict([img_slice, img_slice1], batch_size=1, verbose=1)
                    imgs_test_big = np.where(imgs_test_big > 0.5, 1, 0)


        # np.save(savepath +'patname/' + patname +'test_imgA1.npy', img_npy)
        # np.save(savepath +'patname/' + patname + 'test_imgV1.npy', img_npy1)
        # np.save(savepath +'patname/' + patname + 'test_maskV1.npy', mask_npy)



    # img_aug_npy, mask_aug_npy = Img_AUG(img_npy, mask_npy)
    print('test_A length:',len(img_npy))
    # print('test_V length:', len(img_npy1))
    print('test_mask length:', len(mask_npy))
    np.save(savepath + 'test_plain_img.npy', img_npy)
    # np.save(savepath + 'train_imgV1.npy', img_npy1)
    np.save(savepath + 'test_plain_mask.npy', mask_npy)
    # data1 = pd.DataFrame(pixel)
    # data1.to_csv('./train/NPY/pixels_nums_big.csv')







if __name__ == '__main__':
    filepath = './test/CT_A'
    filepath1 = './test/CT_V'
    maskpath = './test/CT_VSC'
    savepath = './test/NPY/'
    nii_to_image(filepath, filepath1, maskpath, savepath)