# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:16:10 2023

@author: Andres
"""

import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Segmentation_models library
import segmentation_models as sm

from Utils.GetModel3 import GetModel

#%%

with open('input_variables.json','r') as f:
   data = json.load(f)


dictsettings = data["settings"]

main_img_path = dictsettings[0]['img_path']

ModelPath =  dictsettings[1]['ModelPath']
BestModelName = dictsettings[1]['BestModelName'] 
TempModelName = dictsettings[1]['TempModelName']

#%% Parameters

imsize = dictsettings[2]["imsize"]
scale = dictsettings[2]["scale"]
batch_size = dictsettings[2]["batch_size"]
train_model_epochs = dictsettings[2]["train_model_epochs"]
scaleimsize = imsize//scale

ckptmodel_path = ModelPath + TempModelName
bestmodel_path = ModelPath + BestModelName

#%% Verbose -> Model Training Tracking

verbosemodelcheckpoint = dictsettings[4]["verbosemodelcheckpoint"]
verbosemodelfit = dictsettings[4]["verbosemodelfit"]
verboseearlystopping =  dictsettings[4]["verboseearlystopping"]
earlystoppingepochs =  dictsettings[4]["earlystoppingepochs"]

#%%


def TestPool(img_path,mask_path,files,maskfiles,print_metrics_summary,path_model,model):
    
    sm.set_framework('tf.keras')
    sm.framework()

    # BACKBONE = backbone_cnn
    # model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    # model = GetModel()
    
    # print(path_model)
    print("loading weights from: " + path_model)
    model.load_weights(path_model)
    
    dicescore = []
    IoUmetric = []
    Dicemetric = []
    

    for imfile,maskfile in zip(files,maskfiles):
        
        imarray = cv.imread(img_path+imfile[0])
        imarray = imarray/255                   #Normalize [0 to 1]
        
        maskarray=cv.imread(mask_path+maskfile[0]) 
        
        # Resizing image
        imarray_resized = cv.resize(imarray,(scaleimsize,scaleimsize), 
                                interpolation = cv.INTER_AREA)
        
        # Resizing mask
        mask_resized = cv.resize(maskarray,(scaleimsize,scaleimsize), 
                                interpolation = cv.INTER_AREA)
        
        # Prediction
        predmask = np.squeeze(model.predict(np.expand_dims(imarray_resized,axis=0),
                                            verbose=0),
                              axis=0)
        
        # Choose mask (Prediction and groundtruth)
        predmask = np.int16(predmask[:,:,0]>0.5) # Predicted mask
        truemask = np.int16(mask_resized[:,:,2]//255) # Groundtruth mask
        
        # Flattening mask
        predmask_flat = predmask.flatten()
        truemask_flat = truemask.flatten()
        
        # Intersection and Union
        intersection = predmask_flat & truemask_flat
        union = predmask_flat | truemask_flat
        
        if np.sum(truemask_flat)>0:
            
            delta=0.0001
            IoU =np.sum(intersection)/(np.sum(union)+delta)
            Dice = 2*np.sum(intersection)/(np.sum(predmask_flat)+np.sum(truemask_flat)+delta)
    
        Dicemetric.append(Dice)
        IoUmetric.append(IoU)
        
        
    showimage = False
    if showimage==True:
        plt.figure()
        # plt.imshow()
        plt.title('eso')
        
        plt.subplot(1,3,1)
        plt.imshow(imarray)
        plt.axis('off')
        plt.title('Patch')
        
        plt.subplot(1,3,2)
        plt.imshow(truemask,cmap='gray')
        plt.axis('off')
        plt.title('Grountruth')
        
        plt.subplot(1,3,3)
        plt.imshow(predmask, cmap='gray')
        plt.axis('off')
        plt.title('Prediction')
    
    IoU = np.array(IoUmetric)
    meanIoU = np.mean(IoUmetric)
    stdIoU = np.std(IoUmetric)
    
    dicescore = np.array(Dicemetric)
    meandice = np.mean(dicescore)
    stddice = np.std(dicescore)
        
    if print_metrics_summary==True:
        
        print('------------------------')
        print(f'IoU Score: {meanIoU:.3f} +- {stdIoU:.3f}')
        print('------------------------')
        print(f'Dice Score: {meandice:.3f} +- {stddice:.3f}')
        print('------------------------')
    
    return Dicemetric,meandice,stddice,meanIoU,stdIoU