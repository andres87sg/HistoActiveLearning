# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 06:00:14 2023

@author: Andres
"""

# =============================================================================
# 
# Script for testing the segmentation models
#
# =============================================================================

import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

# Segmentation_models library
import segmentation_models as sm
# from Utils.GetModel import GetModel
from Utils.GetModel3 import GetModel


sm.set_framework('tf.keras')
sm.framework()

# path_model = "C:/Users/Andres/Desktop/BestAL_MV_Exp0_DataAugm7.h5"

model = GetModel()
# model.load_weights(path_model)


# path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_TCGA_ChA/MV/'
# destmaskpath = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_TCGA_ChA/MV_SG/'

path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_raw/Testing/MV/'
destmaskpath = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_raw/Testing/MV_SG2/'

imsize = 896
scale = 4
scaleimsize = imsize//scale

files = sorted(os.listdir(path))
# maskfiles = sorted(os.listdir(destmaskpath))
#%%
import tensorflow as tf
tensor = []
for imfile in files:
    # print(num)
    imarray = cv.imread(path+imfile)
    imarray = imarray/255                   #Normalize [0 to 1]
    imarray_resized = cv.resize(imarray,(scaleimsize,scaleimsize), 
                            interpolation = cv.INTER_AREA)
    
    tensor.append(imarray_resized)

kk = tf.convert_to_tensor(
    tensor[:4], dtype=tf.float16, dtype_hint=None, name=None)

#%%
import tensorflow_io as tfio

x = tfio.experimental.color.rgb_to_lab(kk)

L = x[:,:,:,0]
A = x[:,:,:,1]
B = x[:,:,:,2]
a=0

#%%
ll = model.predict(kk)

zz = ll[0,:,:,0]
plt.imshow(zz)
np.min(zz)

#%%
import tensorflow as tf
import tensorflow_io as tfio

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image



#%%

predmask_aux = np.zeros((scaleimsize,scaleimsize,3))

tensor = []
 
# for imfile,maskfile in zip(files,maskfiles):
# for imfile,maskfile in zip(files[14:15],maskfiles[14:15]):
i=0    

for imfile in files:
    # print(num)
    imarray = cv.imread(path+imfile)
    imarray = imarray/255                   #Normalize [0 to 1]
    tensor.append(imarray)

kk = tf.convert_to_tensor(
    tensor[:4], dtype=tf.float16, dtype_hint=None, name=None
)

kk = check_image(kk)


lab = tfio.experimental.color.rgb_to_lab(kk)

#%%
# def rescale_0_1(tensor):
#     tensor = tf.cast(tensor, tf.float64)
#     tensor = (tensor - tf.math.reduce_min(tensor)) * (1 / (tf.math.reduce_max(tensor) - tf.math.reduce_min(tensor)))
#     return tensor

# def rescale_0_1_channel_wise(tensor):
#     num_channels = tf.shape(tensor)[-1]
#     channels = tf.TensorArray(tf.float64, size=num_channels)
#     for channel_idx in tf.range(num_channels):
#         channel = rescale_0_1(tensor[:,:,channel_idx])
#         channels = channels.write(channel_idx, channel)
#     tensor = tf.transpose(channels.stack(), [1,2,0])
#     return tensor

#%%
import tensorflow as tf
import tensorflow.keras as keras

from keras import layers, Model
# Segmentation_models library
import segmentation_models as sm

backbone_cnn = 'efficientnetb0'
sm.set_framework('tf.keras')
sm.framework()
BACKBONE = backbone_cnn

imsize = 896

positive_input = layers.Input(name="Input Layer", 
                              shape=(imsize,imsize,3))

x = tfio.experimental.color.rgb_to_lab(positive_input,)

x = tf.keras.layers.Rescaling(scale=1./255)(x)

# x = rescale_0_1_channel_wise(x)
# ResizeLayerImOr = layers.Resizing(imsize,imsize,
                                  # interpolation='bilinear')(positive_input)

cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)

out = cnn1(x)


model = Model(inputs = positive_input, outputs = x, name="Emb")

#%%

kkz = model.predict(kk)

#%%


#%%
"""
    # maskarray=cv.imread(maskpath+maskfile) 
    
    # Resizing image
    imarray_resized = cv.resize(imarray,(scaleimsize,scaleimsize), 
                            interpolation = cv.INTER_AREA)
    
    # Resizing mask
    # mask_resized = cv.resize(maskarray,(scaleimsize,scaleimsize), 
    #                         interpolation = cv.INTER_AREA)
    
    # Prediction
    predmask = np.squeeze(model.predict(np.expand_dims(imarray_resized,axis=0),verbose=0),
                          axis=0)
    
    predmask = np.int8(np.round(predmask[:,:,0])) # Predicted mask
    
    for i in range(3): predmask_aux[:,:,i] = predmask

    predmask_resized =  np.round(cv.resize(predmask_aux,(imsize,imsize), 
                            interpolation = cv.INTER_LINEAR))*255
    
    
    
    plt.show()
    plt.subplot(1,2,1)
    plt.imshow(imarray)
    plt.title('Patch')
    plt.axis('off')

    
    # plt.subplot(1,3,2)
    # plt.imshow(truemask,cmap='gray')
    # plt.axis('off')
    # plt.title('Grountruth')
    
    plt.subplot(1,2,2)
    plt.imshow(predmask, cmap='gray')
    plt.axis('off')
    plt.title('Prediction')
    
    cv.imwrite(destmaskpath+'SG_'+ imfile, predmask_resized)
"""