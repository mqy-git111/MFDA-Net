import numpy as np
from PIL import Image
import os
import cv2

def convert_png_to_npy(png_file_path, npy_file_path):
    # 读取PNG图像
    # image = Image.open(png_file_path)
    image = cv2.imread(png_file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=2)
    # 将图像转换为NumPy数组
    # image_array = np.array(image)
    # 保存为.npy文件
    np.save(npy_file_path, gray_image)


def convert_mask_to_npy(png_file_path, npy_file_path):
    # 读取PNG图像
    # image = Image.open(png_file_path)
    image = cv2.imread(png_file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=2)
    # 将图像转换为NumPy数组
    # image_array = np.array(image)
    # 保存为.npy文件
    np.save(npy_file_path, gray_image)


# 使用函数转换并保存文件
pathv = "./v"
patha = "./a"
pathmask = "./mask"
savepath_v = "./train_img_v/"
savepath_a = "./train_img_a/"
savepath_mask = "./train_mask_v/"
if os.path.exists(savepath_a) == False:
    os.makedirs(savepath_a)
if os.path.exists(savepath_v) == False:
    os.makedirs(savepath_v)
if os.path.exists(savepath_mask) == False:
    os.makedirs(savepath_mask)
list = os.listdir(pathmask)
list.sort()
train_list = list[:int(len(list) * 0.8)]
test_list = list[int(len(list) * 0.8):]
for name in train_list:
    convert_mask_to_npy(pathmask + '/' + name, savepath_mask +name.replace("png","npy"))
a_list = []
v_list = []
mask_list = []
for name in test_list:
    png_file_path = pathv + '/' + name
    image = cv2.imread(png_file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=2)
    v_list.append(gray_image)

    png_file_path = patha + '/' + name
    image = cv2.imread(png_file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=2)
    a_list.append(gray_image)

    png_file_path = pathmask + '/' + name
    image = cv2.imread(png_file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=2)
    mask_list.append(gray_image)


# 使用numpy.array()函数将列表转换为numpy数组
v = np.array(v_list)
a = np.array(a_list)
mask = np.array(mask_list)
npy_file_path = "../NII_TO_IMG/test1/v.npy"
np.save(npy_file_path, v)
npy_file_path = "../NII_TO_IMG/test1/a.npy"
np.save(npy_file_path, a)
npy_file_path = "../NII_TO_IMG/test1/mask.npy"
np.save(npy_file_path, mask)
