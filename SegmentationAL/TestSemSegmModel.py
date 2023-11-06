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
from Utils.GetModel import GetModel

sm.set_framework('tf.keras')
sm.framework()

# path_model = "C:/Users/Andres/Desktop/BestALModel_PC_08102023_Exp1.h5"
path_model = "D:/GBM_Project/Experiments/CurrentModels/ModelPC.h5"

model = GetModel()
model.load_weights(path_model)

#%%

# path = 'D:/TCGA-GBM_Patches_MV_LAB/'
# path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_TCGA_ChA/MV/'
path = 'D:/TCGA-GBM_Patches_PC_LAB/'
destmaskpath = 'D:/mask_PC/'

savemask = True
# path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_data_augm/Testing/MV/'
# destmaskpath = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_data_augm/Testing/MV_SG2/'

imsize = 1792
scale = 8
scaleimsize = imsize//scale

files = sorted(os.listdir(path))
# maskfiles = sorted(os.listdir(destmaskpath))
#%%
predmask_aux = np.zeros((scaleimsize,scaleimsize,3))
 
# for imfile,maskfile in zip(files,maskfiles):
# for imfile,maskfile in zip(files[14:15],maskfiles[14:15]):
for imfile in files:
    # print(num)
    imarray = cv.imread(path+imfile)
    imarray = imarray/255                   #Normalize [0 to 1]
    
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
    
    if savemask == True:
        cv.imwrite(destmaskpath+'SG_'+ imfile, predmask_resized)
    
    
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
