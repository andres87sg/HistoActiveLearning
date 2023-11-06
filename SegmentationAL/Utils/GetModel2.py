# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:09:23 2023

@author: Andres
"""
import tensorflow as tf
import tensorflow.keras as keras

from keras import layers, Model
# Segmentation_models library
import segmentation_models as sm

def GetModel():
     
    backbone_cnn = 'efficientnetb0'
    sm.set_framework('tf.keras')
    sm.framework()
    BACKBONE = backbone_cnn
    
    imsize = 224
    
    positive_input = layers.Input(name="Input Layer", 
                                  shape=(imsize,imsize,3))
    
    ResizeLayerImOr = layers.Resizing(imsize,imsize,
                                      interpolation='bilinear')(positive_input)
    
    # Upper Left  (UL)
    # Upper Right (UR)
    # Lower Left  (LL)
    # Lower Right (LR)
    
    imsize_subsampled = int(imsize/2)
    
    top_crop = bottom_crop = left_crop = right_crop = int(imsize/2)
    
    CropLayer_UL = layers.Cropping2D(cropping=((0,bottom_crop),(0, right_crop)),
                                    data_format="channels_last")(positive_input)
    
    CropLayer_UR = layers.Cropping2D(cropping=((0,bottom_crop),(left_crop, 0)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer_LL = layers.Cropping2D(cropping=((top_crop, 0),(0, right_crop)),
                                   data_format="channels_last")(positive_input)
    
    CropLayer_LR = layers.Cropping2D(cropping=((top_crop, 0),(left_crop,0)),
                                   data_format="channels_last")(positive_input)
    
    ResizeLayer_UL = layers.Resizing(imsize,imsize,
                                    interpolation='bilinear')(CropLayer_UL)
    
    ResizeLayer_UR = layers.Resizing(imsize,imsize,
                                    interpolation='bilinear')(CropLayer_UR)
    
    ResizeLayer_LL = layers.Resizing(imsize,imsize,
                                    interpolation='bilinear')(CropLayer_LL)
    
    ResizeLayer_LR = layers.Resizing(imsize,imsize,
                                    interpolation='bilinear')(CropLayer_LR)

    UNet = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    
    cnn1 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    cnn2 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    cnn3 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    cnn4 = sm.Unet(BACKBONE, classes=1, activation='sigmoid',encoder_weights=None)
    
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
    
    ResizeLayerOut_UL = tf.keras.layers.Resizing(imsize_subsampled,imsize_subsampled,
                                               interpolation='bilinear')(emb1.output)
    
    ResizeLayerOut_UR = tf.keras.layers.Resizing(imsize_subsampled,imsize_subsampled,
                                               interpolation='bilinear')(emb2.output)
    
    ResizeLayerOut_LL = tf.keras.layers.Resizing(imsize_subsampled,imsize_subsampled,
                                               interpolation='bilinear')(emb3.output)
    
    ResizeLayerOut_LR = tf.keras.layers.Resizing(imsize_subsampled,imsize_subsampled,
                                               interpolation='bilinear')(emb4.output)
    
    z1 = keras.layers.Concatenate(axis=2)([ResizeLayerOut_UL, ResizeLayerOut_UR])
    z2 = keras.layers.Concatenate(axis=2)([ResizeLayerOut_LL, ResizeLayerOut_LR])
    z3 = keras.layers.Concatenate(axis=1)([z1, z2])
    
    x = keras.layers.Add()([emb0(positive_input), z3])
    
    x = tf.keras.layers.Resizing(imsize,imsize,
                                  interpolation='bilinear')(x)
    
    x = layers.Conv2D(64, 1, padding='valid',activation="relu")(x)
    # x = layers.Conv2D(64, 1, padding='valid', activation="relu")(x)
    # x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    # x = layers.Conv2D(32, 1, padding='valid', activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    
    # Embedding model
    
    model = Model(inputs = positive_input, 
                  outputs = out, name="Emb")
     
    
    return model

    
