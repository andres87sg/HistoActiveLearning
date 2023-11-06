import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

# tf.test.gpu_device_name()

from keras.preprocessing.image import ImageDataGenerator

from keras import Input,layers, models
from keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# Segmentation_models libra
import segmentation_models as sm
from segmentation_models import metrics


# imsize = 3584
# scale = 16
# scaleimsize = imsize//scale
# batch_size = 4

# train_model_epochs = 5
# ckptmodel_path ='C:/Users/Andres/Desktop/TempModelPC.h5'
# bestmodel_path = 'C:/Users/Andres/Desktop/BestAL_PC2.h5'
backbone_cnn = 'efficientnetb0'

def createmodel():
      
    sm.set_framework('tf.keras')
    sm.framework()
    
    imsize = 448
    scale = 2
    
    scaleimsize = int(imsize//scale)
    
    downsamplefactor = 2
    downimsize = int(scaleimsize//downsamplefactor)
    
    # def TrainModel(train_generator,steps_train,load_weights):
        
    BACKBONE = backbone_cnn
    
    # define model
    UNet = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn2 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn3 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn4 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    # UNet.load_weights('C:/Users/Andres/Desktop/PretrainedModel.h5')
    # cnn1.load_weights('C:/Users/Andres/Desktop/PretrainedModel.h5')
    # cnn2.load_weights('C:/Users/Andres/Desktop/PretrainedModel.h5')
    # cnn3.load_weights('C:/Users/Andres/Desktop/PretrainedModel.h5')
    # cnn4.load_weights('C:/Users/Andres/Desktop/PretrainedModel.h5')
    
    positive_input = layers.Input(name="Input Layer", 
                                  shape=(scaleimsize,scaleimsize,3))
    
    
    CropLayer1 = layers.Cropping2D(cropping=((0,downimsize),(0, downimsize)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer2 = layers.Cropping2D(cropping=((0, downimsize),(downimsize, 0)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer3 = layers.Cropping2D(cropping=((downimsize, 0),(0, downimsize)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer4 = layers.Cropping2D(cropping=((downimsize, 0),(downimsize, 0)),
                                   data_format="channels_last")(positive_input)
    
    ResizeLayer1 = layers.Resizing(scaleimsize,scaleimsize,
                                   interpolation='bilinear')(CropLayer1)
    
    ResizeLayer2 = layers.Resizing(scaleimsize,scaleimsize,
                                   interpolation='bilinear')(CropLayer2)
    
    ResizeLayer3 = layers.Resizing(scaleimsize,scaleimsize,
                                   interpolation='bilinear')(CropLayer3)
    
    ResizeLayer4 = layers.Resizing(scaleimsize,scaleimsize,
                                   interpolation='bilinear')(CropLayer4)
    
    out1 = cnn1(ResizeLayer1)
    out2 = cnn2(ResizeLayer2)
    out3 = cnn3(ResizeLayer3)
    out4 = cnn4(ResizeLayer4)
    
    emb1 = Model(positive_input,out1,name='Model_UL')
    emb2 = Model(positive_input,out2,name='Model_UR')
    emb3 = Model(positive_input,out3,name='Model_LL')
    emb4 = Model(positive_input,out4,name='Model_LR')
    
    ResizeLayer1out = tf.keras.layers.Resizing(downimsize,downimsize,
                                               interpolation='bilinear')(emb1.output)
    
    ResizeLayer2out = tf.keras.layers.Resizing(downimsize,downimsize,
                                               interpolation='bilinear')(emb2.output)
    
    ResizeLayer3out = tf.keras.layers.Resizing(downimsize,downimsize,
                                               interpolation='bilinear')(emb3.output)
    
    ResizeLayer4out = tf.keras.layers.Resizing(downimsize,downimsize,
                                               interpolation='bilinear')(emb4.output)
    
    z1 = keras.layers.Concatenate(axis=2)([ResizeLayer1out, ResizeLayer2out])
    z2 = keras.layers.Concatenate(axis=2)([ResizeLayer3out, ResizeLayer4out])
    z3 = keras.layers.Concatenate(axis=1)([z1, z2])
    
    x = keras.layers.Add()([UNet(positive_input), z3])
    # x = keras.layers.Concatenate()([UNet(positive_input), z3])
    
    x = layers.Conv2D(64, 1, padding='valid',activation="relu",name='eso')(x)
    x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    x = layers.Conv2D(16, 1, padding='valid', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(
        inputs=positive_input, outputs= out, name="Emb"
    )
    
    return model


"""
def createmodel():

    sm.set_framework('tf.keras')
    sm.framework()
    
    imsize = 448
    
    positive_input = layers.Input(name="Input Layer", 
                                  shape=(imsize,imsize,3))
    
    ResizeLayerImOr = layers.Resizing(448,448,
                                      interpolation='bilinear')(positive_input)
    
    # Upper Left  (UL)
    # Upper Right (UR)
    # Lower Left  (LL)
    # Lower Right (LR)
    
    # bottom_crop = int(imsize/2)
    top_crop = bottom_crop = left_crop = right_crop = int(imsize/2)
    
    # ((top_crop, bottom_crop), (left_crop, right_crop)
    
    
    CropLayer_UL = layers.Cropping2D(cropping=((0,bottom_crop),(0, right_crop)),
                                    data_format="channels_last")(positive_input)
    
    CropLayer_UR = layers.Cropping2D(cropping=((0,bottom_crop),(left_crop, 0)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer_LL = layers.Cropping2D(cropping=((top_crop, 0),(0, right_crop)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer_LR = layers.Cropping2D(cropping=((top_crop, 0),(left_crop,0)),
                                   data_format="channels_last")(positive_input)
    
    ResizeLayer_UL = layers.Resizing(448,448,
                                    interpolation='bilinear')(CropLayer_UL)
    
    ResizeLayer_UR = layers.Resizing(448,448,
                                    interpolation='bilinear')(CropLayer_UR)
    
    ResizeLayer_LL = layers.Resizing(448,448,
                                    interpolation='bilinear')(CropLayer_LL)
    
    ResizeLayer_LR = layers.Resizing(448,448,
                                    interpolation='bilinear')(CropLayer_LR)
    
    BACKBONE = backbone_cnn
    # define model
    UNet = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn2 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn3 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    cnn4 = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    out_Or = UNet(ResizeLayerImOr)
    out_UL = cnn1(ResizeLayer_UL)
    out_UR = cnn2(ResizeLayer_UR)
    out_LL = cnn3(ResizeLayer_LL)
    out_LR = cnn4(ResizeLayer_LR)
    
    emb0 = Model(positive_input,out_Or)
    emb1 = Model(positive_input,out_UL)
    emb2 = Model(positive_input,out_UR)
    emb3 = Model(positive_input,out_LL)
    emb4 = Model(positive_input,out_LR)
    
    ResizeLayerOut_UL = tf.keras.layers.Resizing(224,224,
                                               interpolation='bilinear')(emb1.output)
    
    ResizeLayerOut_UR = tf.keras.layers.Resizing(224,224,
                                               interpolation='bilinear')(emb2.output)
    
    ResizeLayerOut_LL = tf.keras.layers.Resizing(224,224,
                                               interpolation='bilinear')(emb3.output)
    
    ResizeLayerOut_LR = tf.keras.layers.Resizing(224,224,
                                               interpolation='bilinear')(emb4.output)
    
    z1 = keras.layers.Concatenate(axis=2)([ResizeLayerOut_UL, ResizeLayerOut_UR])
    z2 = keras.layers.Concatenate(axis=2)([ResizeLayerOut_LL, ResizeLayerOut_LR])
    z3 = keras.layers.Concatenate(axis=1)([z1, z2])
    
    x = keras.layers.Add()([emb0(positive_input), z3])
    
    x = tf.keras.layers.Resizing(imsize,imsize,
                                 interpolation='bilinear')(x)
    
    x = layers.Conv2D(64, 1, padding='valid',activation="relu")(x)
    x = layers.Conv2D(64, 1, padding='valid', activation="relu")(x)
    x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(
        inputs=positive_input, outputs= out, name="Emb"
    )
      


    
    return model
"""
    
