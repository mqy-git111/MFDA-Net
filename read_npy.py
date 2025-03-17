import numpy as np

path = r"/media/lao/C14D581BDA18EBFA1/xss/Unet/work2/MFDA-Net/NII_TO_IMG/test_1/NPY/test_maskV1.npy"
data = np.load(path)

print('type :', type(data))
print('shape :', data.shape)
print('data :')
print(data)