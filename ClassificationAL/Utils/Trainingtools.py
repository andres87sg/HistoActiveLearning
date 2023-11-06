# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:23:23 2022

@author: Andres
"""

import numpy as np
import sklearn as sklearn
import albumentations as A
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from Utils.GetModel import GetModel

#%% Data Augmentation (
    
def transform(image):
    transform = A.Compose([
        # A.ToFloat(max_value = 255,always_apply=True,p=1.0),
        #A.Downscale(always_apply=False, p=0.5, scale_min=0.25, scale_max=0.75, interpolation=2),
        # A.RGBShift(always_apply=False, p=0.2, r_shift_limit=(-0.1, 0.1), g_shift_limit=(-0.1, 0.1), b_shift_limit=(-0.1, 0.1)),   
        #A.RandomContrast(always_apply=False, p=0.5, limit=(-0.3, 0.3)),
        #A.HueSaturationValue(always_apply=False, p=0.5, hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3)),
       #A.RandomGamma(always_apply=False, p=0.5, gamma_limit=(50, 150), eps=1e-07), 
       # A.Blur(always_apply=False, p=0.5, blur_limit=(2, 4)),
        A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=True),  
        #A.ChannelShuffle(always_apply=False, p=0.5),
        A.Rotate(always_apply=False, p=1.0, limit=(-5, 5), interpolation=4, border_mode=4, value=(0, 0, 0), mask_value=None),
        #A.VerticalFlip(always_apply=False, p=0.5),
        #A.HorizontalFlip(always_apply=False, p=0.5),
        #A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit=(-0.15, 0.15), scale_limit=(-0.3, 0.3), rotate_limit=(-90, 90), interpolation=2, border_mode=3, value=(0, 0, 0), mask_value=None)
        #A.Downscale(always_apply=False, p=0.5, scale_min=0.5, scale_max=0.8999999761581421, interpolation=0),
    ])
    return transform(image=image)['image']



###############

def GeneratorData(BaseTrainData,classes,target_size,batch_size,
                  data_augmentation,shuffle,validation_split):

    traindatagen = ImageDataGenerator(
                                        #Data augmentation
                                      # preprocessing_function=transform,
                                      rescale=1./255,
                                      # fill_mode = 'reflect',
                                      # horizontal_flip=True,
                                      # #vertical_flip=True,
                                      # rotation_range = 30,
                                      validation_split = validation_split
                                      )
    
    validdatagen = ImageDataGenerator(rescale=1./255,
                                      validation_split = validation_split
                                      )
    
    
    TrainImgGenerator = traindatagen.flow_from_dataframe(BaseTrainData,
                                                        x_col='pathfile', 
                                                        classes=classes,
                                                        class_mode='categorical', 
                                                        target_size=target_size,  
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        subset="training",
                                                        seed=1)
    
    ValidImgGenerator = validdatagen.flow_from_dataframe(BaseTrainData,
                                                        x_col='pathfile', 
                                                        classes=classes,
                                                        class_mode='categorical', 
                                                        target_size=target_size,  
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        subset="validation",
                                                        seed=1)
    
    StepsTrain = TrainImgGenerator.n//TrainImgGenerator.batch_size
    StepsValid = ValidImgGenerator.n//ValidImgGenerator.batch_size
    

    return TrainImgGenerator, StepsTrain, ValidImgGenerator, StepsValid

def TestGeneratorData(BaseTrainData,classes,target_size,batch_size,
                  data_augmentation,shuffle):

    # traindatagen = ImageDataGenerator(
    #                                     #Data augmentation
    #                                   preprocessing_function=transform,
    #                                   # rescale=1./255,
    #                                   # fill_mode = 'reflect',
    #                                   # horizontal_flip=True,
    #                                   # #vertical_flip=True,
    #                                   # rotation_range = 30,
    #                                   validation_split = validation_split
                                      # )
    
    validdatagen = ImageDataGenerator(rescale=1./255,
                                      # validation_split = validation_split
                                      )
    
    
    # TrainImgGenerator = traindatagen.flow_from_dataframe(BaseTrainData,
    #                                                     x_col='pathfile', 
    #                                                     classes=classes,
    #                                                     class_mode='categorical', 
    #                                                     target_size=target_size,  
    #                                                     batch_size=batch_size,
    #                                                     shuffle=True,
    #                                                     subset="training",
    #                                                     seed=1)
    
    ValidImgGenerator = validdatagen.flow_from_dataframe(BaseTrainData,
                                                        x_col='pathfile', 
                                                        classes=classes,
                                                        class_mode='categorical', 
                                                        target_size=target_size,  
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        seed=1)
    
    # StepsTrain = TrainImgGenerator.n//TrainImgGenerator.batch_size
    StepsValid = ValidImgGenerator.n//ValidImgGenerator.batch_size
    

    return ValidImgGenerator, StepsValid


def ModelTraining(TrainImgGenerator,StepsTrain,
                   ValidImgGenerator,StepsValid,
                   epochs,ckptmodel_path,verbose):
    
    model = GetModel()
    
    lrop = ReduceLROnPlateau(monitor='loss', 
                             factor=0.1,patience=30, 
                             min_lr=1e-5, verbose=1)
    
    es = EarlyStopping(patience=100,monitor="val_loss",mode='min', verbose=1)
    
    mc = ModelCheckpoint(ckptmodel_path, 
                         monitor='val_loss', 
                         #save_weights_only = True,
                         verbose=2, 
                         save_best_only=True, 
                         mode='min')
    
    model.compile(optimizer = 'Adam',
                  loss = 'categorical_crossentropy',
                  metrics= 'accuracy')
    
    model.fit(TrainImgGenerator,
              steps_per_epoch=StepsTrain,
              validation_data = ValidImgGenerator,
              validation_steps = StepsValid,
              epochs=epochs,
              verbose=verbose,
              callbacks=[es,lrop,mc])
    
    return model


