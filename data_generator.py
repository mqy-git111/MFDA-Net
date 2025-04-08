import keras
from math import ceil
import numpy as np
from data_augmentation import data_aug
import cv2


def mask_edge(x):
    img = x * 255
    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    edge = edge.astype(np.float32)
    edge = edge / 255.0
    edge = edge.reshape((edge.shape[0], edge.shape[1], 1))
    edge = np.where(edge > 0.5, 1, 0)
    return edge

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, img_v_dir, img_a_dir,  mask_dir, batch_size=1, img_size=(512, 512), aug=None,
                 *args, **kwargs):
        """
           self.list_IDs:存放所有需要训练的图片文件名的列表。
           self.labels:记录图片标注的分类信息的pandas.DataFrame数据类型，已经预先给定。
           self.batch_size:每次批量生成，训练的样本大小。
           self.img_size:训练的图片尺寸。
           self.img_dir:图片在电脑中存放的路径。


        """

        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_v_dir = img_v_dir
        self.img_a_dir = img_a_dir
        self.mask_dir = mask_dir
        self.aug = aug
        self.on_epoch_end()


    def __len__(self):
        """
        返回生成器的长度，也就是总共分批生成数据的次数。

        """
        return int(ceil(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """
        该函数返回每次我们需要的经过处理的数据。
         """

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        img_v, img_a, mask = self.__data_generation(list_IDs_temp)
        # img = np.concatenate((img_v, img_a), axis=3)
        # edge_gt = mask_edge(mask)
        return [img_v, img_a], mask


    def on_epoch_end(self):
        """
        该函数将在训练时每一个epoch结束的时候自动执行，在这里是随机打乱索引次序以方便下一batch运行。

        """
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)


    def __data_generation(self, list_IDs_temp):
        """
        给定文件名，生成数据。
        """
        img_v = np.empty((self.batch_size, *self.img_size, 1))
        img_a = np.empty((self.batch_size, *self.img_size, 1))
        mask = np.empty((self.batch_size, *self.img_size, 1))


        def min_max_normalize(data):
            min_val = np.min(data)
            max_val = np.max(data)
            normalized_data = (data - min_val) / (max_val - min_val)
            return normalized_data

        for i, ID in enumerate(list_IDs_temp):
            img_v[i,] = np.load(self.img_v_dir + '/' + ID)
            img_v[i,]= min_max_normalize(img_v[i])
            mask[i,] = np.load(self.mask_dir + '/' + ID)
            ID = ID.replace("v","a")
            img_a[i,] = np.load(self.img_a_dir + '/' + ID)
            img_a[i,] = min_max_normalize(img_a[i])
        mask[mask == 255.0] = 1.0
        return img_v, img_a, mask

