import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Conv2DTranspose, AveragePooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import image
import matplotlib.pyplot as plt
import pickle
from data_generator import DataGenerator
import combo_loss
import loss_functions
import attention_utils
import data_augmentation
import calculate_metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path="./NII_TO_IMG/train/NPY", label_path="./NII_TO_IMG/train/NPY",
                 test_path="./NII_TO_IMG/test/NPY", img_type="bmp"):
        """

        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train_V = np.load(self.data_path + "/train_imgV1.npy")
        imgs_train_A = np.load(self.data_path + "/train_imgA1.npy")
        imgs_mask_train_V = np.load(self.label_path + "/train_maskV1.npy")
        print(imgs_train_V.shape)
        print(imgs_mask_train_V.shape)
        imgs_mask_train_V[imgs_mask_train_V > 0.5] = 1.0
        imgs_mask_train_V[imgs_mask_train_V <= 0.5] = 0.0

        return imgs_train_V, imgs_train_A, imgs_mask_train_V

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test_v_big = np.load(self.test_path + "/test_imgV1_big.npy")
        imgs_test_a_big = np.load(self.test_path + "/test_imgA1_big.npy")
        print(imgs_test_v_big.shape)
        print(imgs_test_a_big.shape)

        imgs_test_v_small = np.load(self.test_path + "/test_imgV1_small.npy")
        imgs_test_a_small = np.load(self.test_path + "/test_imgA1_small.npy")
        print(imgs_test_v_small.shape)
        print(imgs_test_a_small.shape)

        return imgs_test_v_big, imgs_test_a_big, imgs_test_v_small, imgs_test_a_small


class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512):

        self.img_rows = img_rows
        self.img_cols = img_cols
        # self.batch_size = 4
        self.batch_size = 1

    def dataAug(self, batch_imgs_v, batch_mask_v, batch_imgs_a, batch_size, mode, t):

        imgs_v = []
        imgs_a = []
        masks = []
        for i in range(len(batch_imgs_v)):
            img_v, mask_v, img_a = data_augmentation.data_aug(batch_imgs_v[i],batch_mask_v[i],batch_imgs_a[i], 0)
            imgs_v.append(img_v)
            imgs_a.append(img_a)
            masks.append(mask_v)
        return imgs_v, imgs_a, masks

    def get_unet(self):

        #(512,512,1)
        inputs_v = Input((self.img_rows, self.img_cols, 1))
        inputs_a = Input((self.img_rows, self.img_cols, 1))

        conv1_v = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv1_1')(inputs_v)
        conv1_v.trainable = True
        conv1_v = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv1_2')(conv1_v)
        conv1_v.trainable = True

        conv1_a = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv1_1')(inputs_a)
        conv1_a.trainable = True
        conv1_a = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv1_2')(conv1_a)
        conv1_a.trainable = True
        pool1_a = MaxPooling2D(pool_size=(2, 2), name='a_pool1')(conv1_a)

        v_att1 = attention_utils.attention_up_and_concate(conv1_v, conv1_a, name='av1_', train=True, data_format='channels_last')
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(v_att1)

        #(256,256,32)
        conv2_v = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv2_1')(pool1)
        conv2_v.trainable = True
        conv2_v = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv2_2')(conv2_v)
        conv2_v.trainable = True

        conv2_a = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv2_1')(pool1_a)
        conv2_a.trainable = True
        conv2_a = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv2_2')(conv2_a)
        conv2_a.trainable = True
        
        # #
        v_att2 = attention_utils.attention_up_and_concate(conv2_v, conv2_a, name='av_2', train=True, data_format='channels_last')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(v_att2)

        #(128,128,64)
        conv3_v = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv3_1')(pool2)
        conv3_v.trainable = True
        drop3 = Dropout(0.1, name='ddrop3')(conv3_v)
        conv3_v = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv3_2')(drop3)
        conv3_v.trainable = True

        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_v)


        #(64,64,128)
        conv4_v = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv4_1')(pool3)
        conv4_v.trainable = True
        drop4 = Dropout(0.2, name='ddrop4')(conv4_v)
        conv4_v = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv4_2')(drop4)
        conv4_v.trainable = True

        conv4_v = attention_utils.da_attention(conv4_v, name='da_attention_', train=True)
        pool4_v = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_v)

        #(32,32,256)
        conv5_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv5_1')(pool4_v)
        conv5_v.trainable = True
        drop5_v = Dropout(0.2, name='ddrop5')(conv5_v)
        conv5_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv5_2')(drop5_v)
        conv5_v.trainable = True
        pool5_v = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5_v)

        ##(16,16,512)
        conv6_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv6_1')(pool5_v)
        conv6_v.trainable = True
        drop6_v = Dropout(0.2, name='ddrop6')(conv6_v)
        conv6_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv6_2')(drop6_v)
        conv6_v.trainable = True

        # (16,16,1024)
        up5_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up5_1')(
            UpSampling2D(size=(2, 2), name='up5')(conv6_v))
        up5_1.trainable = True
        merge5_1 = concatenate([conv5_v, up5_1], axis=3, name='concat5')
        conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv5_1')(merge5_1)
        conv5_1.trainable = True
        drop5_1v = Dropout(0.2, name='udrop5')(conv5_1)
        conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv5_2')(drop5_1v)
        conv5_1.trainable = True

        # (32,32,512)
        up4_1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_1')(
            UpSampling2D(size=(2, 2), name='up4')(conv5_1)) 
        up4_1.trainable = True
        merge4_1 = concatenate([conv4_v, up4_1], axis=3, name='concat4')
        conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv4_1')(merge4_1)
        conv4_1.trainable = True
        drop4_1v = Dropout(0.2, name='udrop4')(conv4_1)
        conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv4_2')(drop4_1v)
        conv4_1.trainable = True

        up3_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up3_1')(
            UpSampling2D(size=(2, 2), name='up3')(conv4_1))
        up3_1.trainable = True
        merge3_1 = concatenate([conv3_v, up3_1], axis=3, name='concat3')
        conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv3_1')(merge3_1)
        conv3_1.trainable = True
        drop3_1v = Dropout(0.2)(conv3_1)
        conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv3_2')(drop3_1v)
        conv3_1.trainable = True

        up2_1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up2_1')(
            UpSampling2D(size=(2, 2), name='up2')(conv3_1))
        up2_1.trainable = True
        merge2_1 = concatenate([v_att2, up2_1], axis=3, name='concat2')
        conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv2_1')(merge2_1)
        conv2_1.trainable = True
        conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv2_2')(conv2_1)
        conv2_1.trainable = True

        # (256,256,64)
        up1_1 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up1_1')(
            UpSampling2D(size=(2, 2), name='up1')(conv2_1))
        up1_1.trainable = True
        merge1_1 = concatenate([v_att1, up1_1], axis=3, name='concat1')
        conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv1_1')(merge1_1)
        conv1_1.trainable = True
        conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv1_2')(conv1_1)
        conv1_1.trainable = True

        probmap = Conv2D(1, 1, activation='sigmoid', padding='same', name='probmap')(conv1_1)
        probmap.trainable = True

        model = Model(inputs=[inputs_v, inputs_a], output=probmap)

        return model

    def train(self):
        if os.path.exists('./result/da_add_av/cam_av') == False:
            os.makedirs('./result/da_add_av/cam_av')
        model = self.get_unet()
        # multi_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer=Adam(lr=1e-4), loss=combo_loss.Combo_loss, metrics=[combo_loss.Dice])
        model.summary()

        # model.load_weights('./result/da_add_av/cam_av/CAM_net.h5')
        print("got unet")

        print("loading data")
        mydata = dataProcess(self.img_rows, self.img_cols)
        dataset_root_path = "/media/lao/C14D581BDA18EBFA1/xss/Unet/dataset/"
        dataset_root_path = "/media/lao/C14D581BDA18EBFA1/xss/Unet/work2/MFDA-Net-MRI/dataset/"
        img_v_floder = dataset_root_path + "train_img_v"
        img_a_floder = dataset_root_path + "train_img_a"
        mask_floder = dataset_root_path + "train_mask_v"

        img_v_list = os.listdir(img_v_floder)

        count = len(img_v_list)
        print("length of train:", count)
        np.random.seed(10101)
        np.random.shuffle(img_v_list)
        train_imglist = img_v_list[:int(count * 0.005)]
        val_imglist = img_v_list[int(count * 0.995):]
        print("loading data done")

        trainGene = DataGenerator(train_imglist, img_v_floder, img_a_floder, mask_floder,
                                        self.batch_size, img_size=(self.img_rows, self.img_cols), aug=None)
        evalGene = DataGenerator(val_imglist, img_v_floder, img_a_floder, mask_floder,
                                        self.batch_size, img_size=(self.img_rows, self.img_cols), aug=None)


        print("Fitting model...")
        history = model.fit_generator(trainGene,
                                      steps_per_epoch=math.ceil(len(train_imglist)/self.batch_size),
                                      validation_data=evalGene,
                                      validation_steps=math.ceil(len(val_imglist)/self.batch_size),
                                      epochs=100,
                                      callbacks=[
                                          keras.callbacks.ModelCheckpoint('./result/da_add_av/cam_av/CAM_net.h5',
                                                                          monitor='val_loss',
                                                                          verbose=1,
                                                                          save_best_only=True,
                                                                          save_weights_only=True,
                                                                          mode='auto'),
                                          keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0,
                                                                         mode='auto'),
                                          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
                                      ])

        plt.figure()
        plt.title('CAM_net')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/val_loss')
        plt.plot(history.epoch, history.history['loss'], label='train_loss')
        plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig('./result/da_add_av/cam_av/CAM_net')
        with open('./result/da_add_av/cam_av/CAM_net.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        print('predict test data')

        # model = load_model('pre_aug_DA_A_V_att21.hdf5', custom_objects={'CAM':DA_attention.CAM, 'PAM':DA_attention.PAM, 'Combo_loss': combo_loss.Combo_loss})
        imgs_test_v_big, imgs_test_a_big, imgs_test_v_small, img_test_a_small = mydata.load_test_data()

        # multi_model.load_weights('./result/da_add_av/da_add_av.h5')
        model.load_weights('./result/da_add_av/cam_av/CAM_net.h5')
        imgs_mask_test_big = model.predict([imgs_test_v_big, imgs_test_a_big], batch_size=1, verbose=1)
        np.save('./result/da_add_av/cam_av/CAM_net.npy', imgs_mask_test_big)  # 保存test的npy文件位置

    def save_img(self):  # 把测试的结果保存为bmp文件
        print("array to image")
        imgs = np.load('./NII_TO_IMG/test/NPY/pre_DA_A_V_att22.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = image.array_to_img(img)
            img.save("./IMAGEV/test/IMAGE_A/" + str(i) + '.bmp')

    def dice_coef1(self, y_true, y_pred):
        smooth = 1.
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def countdice1(self):
        Dice = []
        dicesum = []
        MASKNames = np.load('./NII_TO_IMG/test/NPY1/test_maskV1_big.npy')  # test对应的mask图片位置
        PreNames = np.load('./result/fuse_way/merge_input/merge_input_big.npy')  # test 预测结果的位置
        for i in range(len(MASKNames)):
            img1 = MASKNames[i]
            img2 = PreNames[i]
            img1 = img1.astype('float32')
            img2 = img2.astype('float32')
            img11 = np.where(img1 > 0.5, 1, 0)  # 大于0.5的为1，其余为0，二值图1为白色，0为黑色
            img22 = np.where(img2 > 0.5, 1, 0)
            dice = self.dice_coef1(img11, img22)
            Dice.append(dice)
            # if dice > 0.1:
            #     dicesum.append(dice)
        aver = np.nanmean(Dice)
        print("dice nums:", len(Dice))
        print("dice:", aver)
        Dice.append(aver)

        data1 = pd.DataFrame(Dice)
        data1.to_csv('./result/merge_input_big.csv')  # 把Dice数据存储到CSV中

        sen, spe, OR, UR = calculate_metrics.Sen_spe(MASKNames, PreNames)
        Sen = pd.DataFrame(sen)
        Spe = pd.DataFrame(spe)
        Sen.to_csv('./result/merge_input_big_sensitivity.csv')
        Spe.to_csv('./result/merge_input_big_specificity.csv')
        os_rate = pd.DataFrame(OR)
        us_rate = pd.DataFrame(UR)
        os_rate.to_csv('./result/merge_input_big_overrate.csv')
        us_rate.to_csv('./result/merge_input_big_underrate.csv')




if __name__ == '__main__':
    myunet = myUnet()
    img = myunet.train()
    # myunet.save_img()
    # myunet.countdice1()
