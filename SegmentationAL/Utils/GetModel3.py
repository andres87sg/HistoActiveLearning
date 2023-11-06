# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 06:40:59 2023

@author: Andres
"""

import tensorflow as tf
# from tf import math
# import tensorflow.keras as keras

from keras import layers, Model
from keras.layers import Normalization
# Segmentation_models library
import segmentation_models as sm

import tensorflow_io as tfio

def GetModel(pretrained=None):
    
    backbone_cnn = 'efficientnetb0'
    sm.set_framework('tf.keras')
    sm.framework()
    
    BACKBONE = backbone_cnn
    
    model = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)

    if pretrained=='imagenet':
        model = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights='imagenet')
    
    
    return model

# model = GetModel()

# model.save_weights('C:/Users/Andres/Desktop/pretrained_imagenet.h5')

#%%

"""
def GetModel():

    backbone_cnn = 'efficientnetb0'
    sm.set_framework('tf.keras')
    sm.framework()
    BACKBONE = backbone_cnn
    
    imsize = 224

    positive_input = layers.Input(name="Input Layer", 
                                  shape=(imsize,imsize,3))

    x = tfio.experimental.color.rgb_to_lab(positive_input)
    
    L = x[:,:,:,0]
    A = x[:,:,:,1]
    B = x[:,:,:,2]
       
    L_out = tf.math.divide(L,100)
    
    A1 = tf.add(A,86)
    A_out = tf.math.divide(A1,184)
    
    B1 = tf.add(B,107)
    B_out = tf.math.divide(B1,202)
    
    L_out = Normalization(axis=-1)(L_out)
    A_out = Normalization(axis=-1)(A_out)
    B_out = Normalization(axis=-1)(B_out)

    
    LAB_L_out = tf.stack([L_out, L_out,L_out],-1)
    LAB_A_out = tf.stack([A_out, A_out,A_out],-1)
    LAB_B_out = tf.stack([B_out, B_out,B_out],-1)
    
    # x = tfio.experimental.color.lab_to_rgb(x)
    
    # x = layers.Normalization(axis=-1)(x)

    # ResizeLayerImOr = layers.Resizing(imsize,imsize,
                                      # interpolation='bilinear')(positive_input)
                                      
    # x = layers.Rescaling(scale=1./255, offset=1)(x)
    # x = layers.Rescaling(scale=1./127.5, offset=-1)(x)

    # pathmodel = 'D:/GBM_Project/Experiments/CurrentModels/BestAL_MV_Exp0_DataAugm7.h5'

    cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    #cnn2 = sm.Unet(BACKBONE, classes=1, activation='softmax',encoder_weights=None)
    #cnn3 = sm.Unet(BACKBONE, classes=1, activation='softmax',encoder_weights=None)
    #cnn4 = sm.Unet(BACKBONE, classes=1, activation='softmax',encoder_weights=None)
    
    path_model_weights ='D:/GBM_Project/Experiments/CurrentModels/BestAL_MV_Exp0_DataAugm7.h5' 
    cnn1.load_weights(path_model_weights)
    #cnn2.load_weights(path_model_weights)
    #cnn3.load_weights(path_model_weights)
    #cnn4.load_weights(path_model_weights)
    # cnn1.layers.load_weights()
    
    # out_L = cnn1(LAB_L_out)
    out_A = cnn1(LAB_A_out)
    #out_B = cnn3(LAB_B_out)
    #out_RGB = cnn4(positive_input)
    
    #out_L = layers.Conv2D(1, 1, padding='valid',activation="linear",kernel_regularizer=('l1_l2'))(out_L)
    #out_A = layers.Conv2D(1, 1, padding='valid',activation="linear",kernel_regularizer=('l1_l2'))(out_A)
    #out_B = layers.Conv2D(1, 1, padding='valid',activation="linear",kernel_regularizer=('l1_l2'))(out_B)
    #out_RGB = layers.Conv2D(1, 1, padding='valid',activation="linear",kernel_regularizer=('l1_l2'))(out_RGB)
    # add = tf.math.add(out_B,out_A)

    
    # x = tf.keras.layers.Concatenate(axis=-1)([
    #                                           # add,
    #                                           out_RGB,
    #                                           out_L,
    #                                           out_A, 
    #                                           out_B
    #                                           ])
    
    # x = tf.math.add(out_L, out_A)
    # x = tf.math.add(x, out_B)
    # x = tf.math.add(x, out_RGB)
    
    #x = layers.BatchNormalization(axis=-1)(x)
    
    # #out_A_B = tf.math.multiply(out_A, out_B)
    
    # #out_A_B = tf.math.add(out_A, out_B)
    
    
    # #x = tf.math.multiply(out_L,out_A_B)
    
    # x = tf.keras.layers.Concatenate(axis=-1)([out_A, out_B])
    
    # #x = tf.keras.layers.Concatenate(axis=-1)([out_L, x])
    
    # # concatted = tf.keras.layers.Concatenate()([out, out2])
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(2, 1, padding='valid',activation="linear")(x)
    # x = layers.BatchNormalization(axis=-1)(x)1
    # x = layers.Conv2D(8, 1, padding='valid',activation="linear")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(4, 1, padding='valid',activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(4, 1, padding='valid',activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(4, 1, padding='valid',activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(1, 1, padding='valid',activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(2, 1, padding='valid',activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(4, 1, padding='valid',activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(2, 1, padding='valid',activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(2, 1, padding='valid',activation="relu")(x)
    # #x = layers.BatchNormalization(axis=-1)(x)
    # #x = layers.Conv2D(16, 1, padding='valid', activation="relu")(x)
    # # x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    # # x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    
    out = layers.Conv2D(1, 1, padding='same',activation="sigmoid")(x)
    # out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs = positive_input, outputs = out, name="Emb")
    
    return model
"""
#%%
"""
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

# Segmentation_models library
import segmentation_models as sm
# from Utils.GetModel import GetModel
# from Utils.GetModel3 import GetModel

sm.set_framework('tf.keras')
sm.framework()

# path_model = "C:/Users/Andres/Desktop/BestAL_MV_Exp0_DataAugm7.h5"

model = GetModel()

path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_raw/Testing/MV/'
destmaskpath = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_raw/Testing/MV_SG2/'

imsize = 896
scale = 4
scaleimsize = imsize//scale

files = sorted(os.listdir(path))
# maskfiles = sorted(os.listdir(destmaskpath))
#]import tensorflow as tf
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

ll = model.predict(kk)
plt.imshow(ll[0,:,:,0])

print(f'Max: {ll.max()}')
print(f'Min: {ll.min()}')

#%%
#backbone_cnn = 'efficientnetb0'
#sm.set_framework('tf.keras')
#sm.framework()
#BACKBONE = backbone_cnn
#cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None,)

#%%
from keras.utils.vis_utils import plot_model

model.summary()
file = 'C:/Users/Andres/Documents/GitHub/HistoSegmentation/SegmentationAL/Model_Diagram.png'
#if not os.path.isfile(''):
plot_model(model, to_file=file, show_shapes=True, show_layer_names=True, expand_nested=False)

# return model

#%%

imsize = 224

positive_input = layers.Input(name="Input Layer", 
                              shape=(imsize,imsize,3))

x = tfio.experimental.color.rgb_to_lab(positive_input)

L = x[:,:,:,0]
A = x[:,:,:,1]
B = x[:,:,:,2]



L_out = tensorflow.math.divide(L,100)

A1 = tensorflow.add(A,86)
A_out = tensorflow.math.divide(A1,184)

B1 = tensorflow.add(B,107)
B_out = tensorflow.math.divide(B1,202)

x = tf.stack([L_out, A_out,B_out],-1)

model = Model(inputs = positive_input, outputs = x, name="Emb")

#%%
nn = model.predict(kk)
np.shape(nn)

#%%
zz = tf.add(ll,86)
zz2 = tf.math.divide(zz,184)
zz3 = tf.stack([zz2, zz2,zz2],-1)

"""