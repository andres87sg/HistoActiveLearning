# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:42:01 2022

@author: Andres
"""

import json
#import tensorflow as tf
#import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator


with open('input_variables.json','r') as f:
   data = json.load(f)

# print(data)

dictsettings = data["settings"]
imsize = dictsettings[2]["imsize"]
scale = dictsettings[2]["scale"]
scaleimsize = imsize//scale
batch_size = dictsettings[2]["batch_size"]

train_model_epochs = dictsettings[2]["train_model_epochs"]


def ImGenTrain(img_path,mask_path,train_img_df,train_mask_df,validation_split,DataAugm):
    
    target_size = (scaleimsize,scaleimsize)

    if(DataAugm==0):    
        print('Data Augmentantion OFF')
        image_datagen = ImageDataGenerator(rescale=1./255,
                                           validation_split = validation_split,
                                           )

    if(DataAugm==1):   
        print('Data Augmentantion ON')
        image_datagen = ImageDataGenerator(
                                           rescale=1./255,
                                           # rotation_range=45,
                                           fill_mode='reflect',
                                           zoom_range=[1-0.3,1+0.3],
                                           # zoom_range=0.1,
                                            # brightness_range=(0.5,1.0),
                                           # width_shift_range=0.01,
                                           # height_shift_range=0.01,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           validation_split = validation_split,
                                          )
        
        mask_datagen = ImageDataGenerator(rescale=1./255,
                                           # rotation_range=45,
                                           fill_mode='reflect',
                                           zoom_range=[1-0.3,1+0.3],
                                           # zoom_range=0.1,
                                           # brightness_range=(0.5,1.0),
                                           # width_shift_range=0.01,
                                           # height_shift_range=0.01,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           validation_split = validation_split,
                                           )
        
    image_datagen_val = ImageDataGenerator(rescale=1./255,
                                           validation_split = validation_split,
                                           )
    
    
    image_generator = image_datagen.flow_from_dataframe(
        train_img_df, directory=img_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        subset="training", shuffle=True,
        seed=1)
    
    mask_generator = mask_datagen.flow_from_dataframe(
        train_mask_df, directory=mask_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        subset="training", shuffle=True,
        seed=1)
    
    val_image_generator = image_datagen_val.flow_from_dataframe(
        train_img_df, directory=img_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        subset="validation",
        seed=1)
    
    val_mask_generator = image_datagen_val.flow_from_dataframe(
        train_mask_df, directory=mask_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        subset="validation",
        seed=1)
    
    return image_generator, mask_generator,val_image_generator, val_mask_generator


def ImGenTest(img_path,mask_path,train_img_df,train_mask_df,DataAugm):
    
    target_size = (scaleimsize,scaleimsize)
    
    if(DataAugm==0):    
        print('Data Augmentantion OFF')
        image_datagen = ImageDataGenerator(rescale=1./255)

    if(DataAugm==1):   
        print('Data Augmentantion ON')
        image_datagen = ImageDataGenerator(
                                           rescale=1./255,
                                           # rotation_range=45,
                                           fill_mode='reflect',
                                           #zoom_range=0.2,
                                           zoom_range=0.3,
                                           # width_shift_range=0.01,
                                           # height_shift_range=0.01,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                          )
    
    
    image_generator = image_datagen.flow_from_dataframe(
        train_img_df, directory=img_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        seed=1)
    
    mask_generator = image_datagen.flow_from_dataframe(
        train_mask_df, directory=mask_path, x_col='filename', classes=None,
        class_mode=None, target_size=target_size,  batch_size=batch_size,
        seed=1)
    
    return image_generator, mask_generator
