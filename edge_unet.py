import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
from keras.layers.core import Lambda
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, GlobalAveragePooling2D, Dense
from keras.layers import add, concatenate, multiply, Reshape, AveragePooling2D, subtract, Concatenate
from keras.layers import Layer
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import image
import matplotlib.pyplot as plt
import pickle
import combo_loss
from loss_functions import cross_entropy_loss_RCF, deta_loss, suface_loss
import my_attention
from sklearn.model_selection import train_test_split
import sen_spe
import cv2
import shape_stream_layers
from datasets import DataGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def mask_edge(x):
    data = []
    for i in range(x.shape[0]):
        img = x[i] * 255
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        edge = edge.astype(np.float32)
        edge = edge/255.0
        edge = edge.reshape((edge.shape[0], edge.shape[1], 1))
        edge = np.where(edge > 0.5, 1, 0)
        data.append(edge)
    return data

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

def resize_to(x, target_t, name, train=True):
    channels = x.get_shape().as_list()[3]
    scale = target_t.get_shape().as_list()[1]//x.get_shape().as_list()[1]
    x = Conv2DTranspose(filters=channels,
                        kernel_size=scale,
                        strides=scale,
                        name=name+'_trans_conv',
                        trainable=train)(x)
    return x


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
        imgs_test_v = np.load(self.test_path + "/test_imgV1_big.npy")
        imgs_test_a = np.load(self.test_path + "/test_imgA1_big.npy")
        print(imgs_test_v.shape)
        print(imgs_test_a.shape)

        return imgs_test_v, imgs_test_a


class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = 4

    def get_unet(self):

        #(512,512,1)
        inputs_v = Input((self.img_rows, self.img_cols, 1))
        inputs_a = Input((self.img_rows, self.img_cols, 1))

        conv1_v = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv1_1', trainable=True)(inputs_v)
        conv1_v = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv1_2', trainable=True)(conv1_v)

        conv1_a = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv1_1', trainable=True)(inputs_a)
        conv1_a = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv1_2', trainable=True)(conv1_a)
        pool1_a = MaxPooling2D(pool_size=(2, 2), name='a_pool1')(conv1_a)

        v_att1 = my_attention.attention_up_and_concate(conv1_v, conv1_a, name='av1_', train=True, data_format='channels_last')
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(v_att1)

        conv2_v = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv2_1', trainable=True)(pool1)
        conv2_v = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dv_conv2_2', trainable=True)(conv2_v)

        conv2_a = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv2_1', trainable=True)(pool1_a)
        conv2_a = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='da_conv2_2', trainable=True)(conv2_a)

        v_att2 = my_attention.attention_up_and_concate(conv2_v, conv2_a, name='av_2', train=True, data_format='channels_last')
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(v_att2)

        conv3_v = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv3_1', trainable=True)(pool2)
        drop3 = Dropout(0.1, name='ddrop3')(conv3_v)
        conv3_v = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv3_2', trainable=True)(drop3)
        pool3_v = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_v)

        conv4_v = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv4_1', trainable=True)(pool3_v)
        drop4 = Dropout(0.2, name='ddrop4')(conv4_v)
        conv4_v = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv4_2', trainable=True)(drop4)
        conv4_v = my_attention.da_attention(conv4_v, name='da_attention_', train=True)
        pool4_v = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_v)

        conv5_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv5_1', trainable=True)(pool4_v)
        drop5_v = Dropout(0.2, name='ddrop5')(conv5_v)
        conv5_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv5_2', trainable=True)(drop5_v)
        pool5_v = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5_v)

        conv6_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv6_1', trainable=True)(pool5_v)
        drop6_v = Dropout(0.2, name='ddrop6')(conv6_v)
        conv6_v = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='dconv6_2', trainable=True)(drop6_v)

        # ## edge stream ##
        # s1 = Conv2D(filters=16, kernel_size=1, name='s1', trainable=True)(v_att1)
        # s2 = Conv2D(filters=16, kernel_size=1, name='s2', trainable=True)(v_att2)
        # s3 = Conv2D(filters=16, kernel_size=1, name='s3', trainable=True)(conv3_v)
        # s4 = Conv2D(filters=16, kernel_size=1, name='s4', trainable=True)(conv4_v)
        #
        # shortcut1 = s1
        # s1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res1_conv1', trainable=True)(s1)
        # s1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res1_conv2', trainable=True)(s1)
        # x = add([shortcut1, s1], name='res1_add')
        # s2 = resize_to(s2, target_t=x, name='s2_resize', train=True)
        # # gated_conv_1
        # features1 = concatenate([x, s2], axis=-1, name='gate1_concat')
        # alpha1 = Conv2D(filters=16,
        #                 kernel_size=3,
        #                 activation='relu',
        #                 padding='same',
        #                 kernel_initializer='he_normal',
        #                 name='alpha1_conv',
        #                 trainable=True)(features1)
        # alpha1 = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='alpha1', trainable=True)(alpha1)
        # gated1 = multiply([x, alpha1], name='gate1_multi')
        # gated1 = add([x, gated1], name='gate1_add')
        # gated1 = Conv2D(filters=16, kernel_size=1, name='gate_1', trainable=True)(gated1)
        #
        # shortcut2 = gated1
        # s2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res2_conv1', trainable=True)(gated1)
        # s2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res2_conv2', trainable=True)(s2)
        # x = add([shortcut2, s2], name='res2_add')
        # s3 = resize_to(s3, target_t=x, name='s3_resize', train=True)
        # # gated_conv_1
        # features2 = concatenate([x, s3], axis=-1, name='gate2_concat')
        # alpha2 = Conv2D(filters=16,
        #                 kernel_size=3,
        #                 activation='relu',
        #                 padding='same',
        #                 kernel_initializer='he_normal',
        #                 trainable=True,
        #                 name='alpha2_conv')(features2)
        # alpha2 = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='alpha2', trainable=True)(alpha2)
        # gated2 = multiply([x, alpha2], name='gate2_multi')
        # gated2 = add([x, gated2], name='gate2_add')
        # gated2 = Conv2D(filters=16, kernel_size=1, name='gate_2', trainable=True)(gated2)
        #
        # shortcut3 = gated2
        # s3 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res3_conv1', trainable=True)(gated2)
        # s3 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='res3_conv2', trainable=True)(s3)
        # x = add([shortcut3, s3], name='res3_add')
        # s4 = resize_to(s4, target_t=x, name='s4_resize', train=True)
        # # gated_conv_1
        # features3 = concatenate([x, s4], axis=-1, name='gate3_concat')
        # alpha3 = Conv2D(filters=16,
        #                 kernel_size=3,
        #                 activation='relu',
        #                 padding='same',
        #                 kernel_initializer='he_normal',
        #                 trainable = True,
        #                 name='alpha3_conv')(features3)
        # alpha3 = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='alpha3', trainable=True)(alpha3)
        # gated3 = multiply([x, alpha3], name='gate3_multi')
        # gated3 = add([x, gated3], name='gate3_add')
        # gated3 = Conv2D(filters=16, kernel_size=1, name='gate_3', trainable=True)(gated3)
        #
        # edge_out = Conv2D(1, 1, activation='sigmoid', padding='same', name='edge_out', trainable=True)(gated3)
        #
        # edge_conv_4 = Conv2D(filters=256, kernel_size=1, name='edge_conv')(
        #     MaxPooling2D(pool_size=(8, 8),name='edge_down')(edge_out))

        # (16,16,1024)
        up5_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up5_1', trainable=True)(
            UpSampling2D(size=(2, 2), name='up5')(conv6_v))
        merge5_1 = concatenate([conv5_v, up5_1], axis=3, name='concat5')
        conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv5_1', trainable=True)(merge5_1)
        drop5_1v = Dropout(0.2, name='udrop5')(conv5_1)
        conv5_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv5_2', trainable=True)(drop5_1v)

        # (32,32,512)
        up4_1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up4_1', trainable=True)(
            UpSampling2D(size=(2, 2), name='up4')(conv5_1))  # 这一句的意义是啥？论文结构是上采样直接与上一次的卷积结果进行拼接
        merge4_1 = concatenate([conv4_v, up4_1], axis=3, name='concat4')
        conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv4_1', trainable=True)(merge4_1)
        drop4_1v = Dropout(0.2, name='udrop4')(conv4_1)
        conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv4_2', trainable=True)(drop4_1v)

        up3_1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up3_1', trainable=True)(
            UpSampling2D(size=(2, 2), name='up3')(conv4_1))
        merge3_1 = concatenate([conv3_v, up3_1], axis=3, name='concat3n')
        conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv3_1', trainable=True)(merge3_1)
        drop3_1v = Dropout(0.2, name='udrop3')(conv3_1)
        conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv3_2', trainable=True)(drop3_1v)

        up2_1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up2_1', trainable=True)(
            UpSampling2D(size=(2, 2), name='up2')(conv3_1))
        merge2_1 = concatenate([v_att2, up2_1], axis=3, name='concat2')
        conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv2_1', trainable=True)(merge2_1)
        conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv2_2', trainable=True)(conv2_1)

        # (256,256,64)
        up1_1 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up1_1', trainable=True)(
            UpSampling2D(size=(2, 2), name='up1n')(conv2_1))
        merge1_1 = concatenate([v_att1, up1_1], axis=3, name='concat1n')
        conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv1_1', trainable=True)(merge1_1)
        conv1_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='uconv1_2', trainable=True)(conv1_1)
        mask_pred = Conv2D(1, 1, activation='sigmoid', padding='same', name='probmap', trainable=True)(conv1_1)


        pool_m = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same', name='pool_m')(mask_pred)
        deta_m = subtract([mask_pred, pool_m], name='sub_m')
        deta_m = Lambda(lambda x: K.abs(x), name='deta_m')(deta_m)
        # edge_fuse = concatenate([edge_out, deta_m], axis=-1, name='edge_detam_concat')
        # edge_fuse = Conv2D(1, 1, activation='sigmoid', padding='same', name='edge_fuse', trainable=True)(edge_fuse)
        # edge_fuse.trainable = True

        model = Model(inputs=[inputs_v, inputs_a], outputs=[mask_pred, deta_m])
        # model = Model(inputs=[inputs_v, inputs_a], outputs=[edge_out, probmap])
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'probmap':combo_loss.Combo_loss,
                            'deta_m':combo_loss.Dice_Loss,
                            # 'deta_m':keras.losses.mean_squared_error
                            },
                      loss_weights={
                          'probmap': 1.,
                          'deta_m': 0.5,
                          # 'edge_out': 50
                      },
                      metrics={
                          'probmap':combo_loss.Dice})

        return model



    def train(self):

        print("got unet")
        model = self.get_unet()
        print('trainable')
        for x in model.trainable_weights:
            print(x.name)
        print('\n')
        print('non_trainble')
        for x in model.non_trainable_weights:
            print(x.name)
        print('\n')


        # model.load_weights('./result/edge_seg/by_name_da_add_av.h5', by_name=True)
        model.summary()
        plot_model(model, to_file='./result/edge_seg/edge_seg_sample.png', show_shapes=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint('./result/edge_seg/weights/1epoch{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5', monitor='val_loss', verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0,
                                          mode='auto'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
            ]

        ##  loading datasets  ##
        dataset_root_path = "/media/lao/C14D581BDA18EBFA1/xss/Unet/dataset/"
        img_v_floder = dataset_root_path + "train_img_v"
        img_a_floder = dataset_root_path + "train_img_a"
        mask_floder = dataset_root_path + "train_mask_v"

        img_v_list = os.listdir(img_v_floder)

        count = len(img_v_list)
        print("length of train:", count)
        np.random.seed(10101)
        np.random.shuffle(img_v_list)
        train_imglist = img_v_list[:int(count * 0.8)]
        val_imglist = img_v_list[int(count * 0.8):]

        ##  data generater  ##
        trainGene = DataGenerator(train_imglist, img_v_floder, img_a_floder, mask_floder,
                                  self.batch_size, img_size=(self.img_rows, self.img_cols), aug=None)
        evalGene = DataGenerator(val_imglist, img_v_floder, img_a_floder, mask_floder,
                                 self.batch_size, img_size=(self.img_rows, self.img_cols), aug=None)

        history = model.fit_generator(trainGene,
                                      initial_epoch=0,
                                      steps_per_epoch=math.ceil(len(train_imglist)/self.batch_size),
                                      validation_data=evalGene,
                                      validation_steps=math.ceil(len(val_imglist)/self.batch_size),
                                      epochs=100,
                                      callbacks=callbacks)

        plt.figure()
        plt.title('edge_seg_net')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/val_loss')
        plt.plot(history.epoch, history.history['loss'], label='train_loss')
        plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig('./result/edge_seg/edge_seg_net')
        with open('./result/edge_seg/edge_seg_net.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        print('predict test data')

        # model = load_model('canny_unet.hdf5', custom_objects={'Combo_loss': combo_loss.Combo_loss})
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_test_v, imgs_test_a = mydata.load_test_data()

        # model.load_weights('./result/edge_seg/weights/1epoch004_loss0.260_val_loss0.267.h5')

        imgs_mask_test_big = model.predict([imgs_test_v, imgs_test_a], batch_size=1, verbose=1)
        np.save('./result/edge_seg/mask_seg_net_big.npy', imgs_mask_test_big[0])  # 保存test的npy文件位置
        np.save('./result/edge_seg/edge_seg_net_big.npy', imgs_mask_test_big[1])

    def save_img(self):  # 把测试的结果保存为bmp文件
        print("array to image")
        # masks = np.load('./NII_TO_IMG/test/NPY/test_maskV1.npy')
        imgs = np.load('./NII_TO_IMG/test/NPY/pre_A_V_att2_non2.npy')
        # pres = np.load('./IMAGEV/test/NPY_RESULT/dual_av1.npy')  # test的npy文件位置
        for i in range(imgs.shape[0]):
            # pre = pres[i]
            # pre = np.where(pre > 0.5, 1, 0)
            # pre = image.array_to_img(pre)
            # mask = masks[i]
            img = imgs[i]
            # mask = image.array_to_img(mask)
            img = image.array_to_img(img)
            # pre.save("./IMAGEV/test/dual_av1/" + str(i) + '.bmp')  # test结果另存一个文件夹
            # mask.save("./IMAGEV/test/MASK/" + str(i) + '.bmp')
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
        MASKNames = np.load('./NII_TO_IMG/test/NPY/test_maskV1_big.npy')  # test对应的mask图片位置
        # MASKNames = np.load('./IMAGEV/test/NPY_RESULT/V_eval.npy')
        PreNames = np.load('./result/edge_seg/mask_seg_net_big.npy')  # test 预测结果的位置
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
        print("dice:", aver)
        Dice.append(aver)

        data1 = pd.DataFrame(Dice)
        data1.to_csv('./result/edge_seg/mask_seg_net_big.csv')  # 把Dice数据存储到CSV中

        sen, spe, OR, UR = sen_spe.Sen_spe(MASKNames, PreNames)
        Sen = pd.DataFrame(sen)
        Spe = pd.DataFrame(spe)
        Sen.to_csv('./result/edge_seg/mask_seg_net_big_sensitivity.csv')
        Spe.to_csv('./result/edge_seg/mask_seg_net_big_specificity.csv')
        os_rate = pd.DataFrame(OR)
        us_rate = pd.DataFrame(UR)
        os_rate.to_csv('./result/edge_seg/mask_seg_net_big_overrate.csv')
        us_rate.to_csv('./result/edge_seg/mask_seg_net_big_underrate.csv')




if __name__ == '__main__':
    myunet = myUnet()
    img = myunet.train()
    # myunet.save_img()
    myunet.countdice1()
