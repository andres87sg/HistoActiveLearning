# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:23:20 2023

@author: Andres
"""

import json
import math
# Segmentation_models library
import segmentation_models as sm
from segmentation_models import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Utils.createmodel import createmodel


# from segmentation_models.losses import BinaryCELoss

with open('input_variables.json','r') as f:
   data = json.load(f)


dictsettings = data["settings"]

main_img_path = dictsettings[0]['img_path']
ModelPath =  dictsettings[1]['ModelPath']
BestModelName = dictsettings[1]['BestModelName'] 
TempModelName = dictsettings[1]['TempModelName']

ckptmodel_path = ModelPath + TempModelName
bestmodel_path = ModelPath + BestModelName

#%% Verbose -> Model Training Tracking

verbosemodelcheckpoint = dictsettings[4]["verbosemodelcheckpoint"]
verbosemodelfit = dictsettings[4]["verbosemodelfit"]
verboseearlystopping =  dictsettings[4]["verboseearlystopping"]
earlystoppingepochs =  dictsettings[4]["earlystoppingepochs"]
train_model_epochs = dictsettings[2]["train_model_epochs"]

backbone_cnn = 'efficientnetb0'

#%%

sm.set_framework('tf.keras')
sm.framework()

def TrainModel(train_generator,steps_train,valid_generator,steps_valid,load_weights,model_path):

    model = createmodel()
    
    if load_weights==1:
        print('Loading Weigths')
        # model.load_weights(ckptmodel_path)
        model.load_weights(model_path)
        
    if load_weights==2:
        print('Loading Pretrained Weigths')
        model.layers[9].load_weights('/mnt/rstor/CSE_BME_CCIPD/home/asg143/pretrainedmodel.h5')
        #model.layers[10].load_weights('/mnt/rstor/CSE_BME_CCIPD/home/asg143/pretrainedmodel.h5')
        model.layers[11].load_weights('/mnt/rstor/CSE_BME_CCIPD/home/asg143/pretrainedmodel.h5')
        #model.layers[12].load_weights('/mnt/rstor/CSE_BME_CCIPD/home/asg143/pretrainedmodel.h5')
        #model.layers[19].load_weights('/mnt/rstor/CSE_BME_CCIPD/home/asg143/pretrainedmodel.h5')
        
    def step_decay(epoch):
    	initial_lrate = 1e-4
    	drop = 0.1
    	epochs_drop = 3
    	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    	return lrate
    
    # lr = LearningRateScheduler(step_decay)
    
    lrop = ReduceLROnPlateau(monitor='loss', 
                             factor=0.1,patience=30, 
                             min_lr=1e-5, verbose=0)
    
    es = EarlyStopping(patience = earlystoppingepochs,
                       monitor = "loss",
                       mode = 'min',
                       verbose = verboseearlystopping)
    
    
    
    mc = ModelCheckpoint(ckptmodel_path, 
                         monitor='loss', 
                         save_weights_only = True,
                         verbose=verbosemodelcheckpoint, 
                         save_best_only=True, 
                         mode='min')
    
    # model.compile('Adam',loss=[sm.losses.JaccardLoss(),sm.losses.BinaryCELoss()],
    #                metrics=[metrics.iou_score],)
    
    # model.compile('Adam',loss=[sm.losses.JaccardLoss(per_image=False,smooth=1e-05),
    #                            sm.losses.BinaryFocalLoss(alpha=0.25,gamma=2)
    #                            sm.losses.CategoricalCELoss()],
                  
    model.compile('Adam',loss=[sm.losses.JaccardLoss(per_image=False,smooth=1e-05),
                               sm.losses.BinaryCELoss()
                               ],
                  
    metrics=[metrics.IOUScore(smooth=1e-05),
             metrics.FScore(beta=1,smooth=1e-05)],)
    
    # model.compile('Adam',loss=[sm.losses.DiceLoss()],
                  # metrics=[metrics.iou_score],)
    
    
    epochs = train_model_epochs
    
    model.fit(train_generator,
              steps_per_epoch=steps_train,
              validation_data=valid_generator,
              validation_steps=steps_valid ,
              epochs=epochs,
              verbose=verbosemodelfit,
              # callbacks=[es,lrop]
              callbacks=[es,lrop,mc]
              )
        
    return model
