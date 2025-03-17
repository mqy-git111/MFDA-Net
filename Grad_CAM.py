import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
from keras.layers import Lambda, Permute
from keras import backend as K
from keras import Model
from keras.models import load_model
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.preprocessing.image import image
import tensorflow as tf
import numpy as np
import cv2
import combo_loss
from A_V_unet import myUnet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def normal_permute(x, pattern):
    return K.permute_dimensions(x, pattern)

# def output_heatmap(model, last_conv_layer_name, imgs_test_v):
def output_heatmap(model, last_conv_layer_name, imgs_test_v, imgs_test_a):

    # preds = model.predict([imgs_test_v, imgs_test_a], batch_size=1, verbose=1)
    # index = np.argmax(preds[0])
    # print('index:%s'%index)
    # target_output = preds
    # class_idx = np.argmax(preds[0])
    class_output = model.output[0]
    # class_output1 = model.output

    # last_conv_layer_model = Model(input=model.input, output=model.get_layer(last_conv_layer_name).output)
    # last_conv_layer = last_conv_layer_model.predict([imgs_test_v, imgs_test_a])
    conv_layer = model.get_layer(last_conv_layer_name)
    # target_output = np.transpose(target_output, (3, 1, 2, 0))
    # last_conv_layer = np.transpose(last_conv_layer, (3, 1, 2, 0))

    grads = K.gradients(class_output, conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.layers[0].input, model.layers[1].input], [pooled_grads, conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([imgs_test_v, imgs_test_a])
    # iterate = K.function([model.layers[0].input], [pooled_grads, conv_layer.output[0]])
    # pooled_grads_value, conv_layer_output_value = iterate([imgs_test_v])

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis = -1)
    heatmap = cv2.resize(heatmap, (imgs_test_v.shape[1], imgs_test_v.shape[2]), cv2.INTER_LINEAR)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    print(heatmap.shape)
    return heatmap


# model = load_model('A_V_att2.hdf5', custom_objects={'Combo_loss': combo_loss.Combo_loss})
a = myUnet()
model = a.get_unet()
# multi_model = multi_gpu_model(model, gpus=2)
# multi_model.compile(optimizer=Adam(lr=1e-4), loss=combo_loss.Combo_loss, metrics=[combo_loss.Dice])
# multi_model.load_weights('./result/path_4_add/multi_att_gate.h5')
# model.save_weights('./result/path_4_add/multi_att_gate_1.h5')
model.load_weights('/media/lao/C14D581BDA18EBFA1/xss/Unet/work2/MFDA-Net/result/da_add_av/block_2/da_add_av.h5')
# plot_model(model, to_file='./result/baseline/unet_5_sample.png', show_shapes=True)
# model.summary()
print(model)
imgs_test_v = np.load('/media/lao/C14D581BDA18EBFA1/xss/Unet/dataset/train_img_v/v_707.npy')
imgs_test_a = np.load('/media/lao/C14D581BDA18EBFA1/xss/Unet/dataset/train_img_a/a_707.npy')
img_v = np.empty((1, 512, 512, 1))
img_a = np.empty((1, 512, 512, 1))
img_v[0,] = imgs_test_v
img_a[0,] = imgs_test_a
# imgs_test_v = imgs_test_v[105:106]
# imgs_test_a = imgs_test_a[105:106]
# imgs_test_v = image.array_to_img(imgs_test_v)
# imgs_test_a = image.array_to_img(imgs_test_a)
# # img = img.astype('float32')
# # img = img / 255.0 * 2 - 1
# imgs_test_v = np.expand_dims(imgs_test_v, 0)
# imgs_test_a = np.expand_dims(imgs_test_a, 0)
    # img = image.array_to_img(img)
img = np.uint8(img_v[0] * 255)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# num = (2, 4, 8, 10, 12, 16, 18, 20, 28, 30, 33, 36, 39, 42, 45, 46)
# for i in range(len(num)):
#     layer_name = 'conv2d_' + str(num[i])
#     # heatmap = output_heatmap(model, layer_name, imgs_test_v)
#     heatmap = output_heatmap(model, layer_name, imgs_test_v, imgs_test_a)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#     cv2.imwrite('./result/da_add_av/dconv4_2/da_attention_out/av_2av_fuse' + layer_name + '_105.png', superimposed_img)

layer_name = 'da_conv2_2'
# heatmap = output_heatmap(model, layer_name, imgs_test_v)
heatmap = output_heatmap(model, layer_name, img_v, img_a)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# img = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
superimposed_img = cv2.addWeighted(img, 0, heatmap, 1, 0)
cv2.imwrite('./result/da_add_av/block_2/gam_mqy/a/' + layer_name + '_707_1.png', superimposed_img)

layer_name = 'dv_conv2_2'
# heatmap = output_heatmap(model, layer_name, imgs_test_v)
heatmap = output_heatmap(model, layer_name, img_v, img_a)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# img = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
superimposed_img = cv2.addWeighted(img, 0, heatmap, 1, 0)
cv2.imwrite('./result/da_add_av/block_2/gam_mqy/v/' + layer_name + '_707_1.png', superimposed_img)

layer_name = 'av_2av_fuse'
# heatmap = output_heatmap(model, layer_name, imgs_test_v)
heatmap = output_heatmap(model, layer_name, img_v, img_a)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# img = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
superimposed_img = cv2.addWeighted(img, 0, heatmap, 1, 0)
cv2.imwrite('./result/da_add_av/block_2/gam_mqy/av/' + layer_name + '_707_1.png', superimposed_img)



