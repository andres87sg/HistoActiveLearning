# =============================================================================
# 
# Reference
# https://github.com/qubvel/segmentation_models
# =============================================================================

import json
import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from keras.preprocessing.image import ImageDataGenerator

from keras import Input,layers, models
from keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# Segmentation_models library
import segmentation_models as sm
from segmentation_models import metrics

from createmodel2 import createmodel
from GetImgSubsetGenerator import ImGenTrain, ImGenTest

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%

with open('input_variables.json','r') as f:
   data = json.load(f)

# print(data)

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

backbone_cnn = 'efficientnetb0'
    
# Query pool size (percentage)
query_perc = dictsettings[3]["query_perc"]

# Initial Training size percentage
train_perc = dictsettings[3]["train_perc"]

# Validation Split
validation_split = dictsettings[3]["validation_split"]

# Training, validation and testing images 
img_path = main_img_path + dictsettings[0]['train_img_path']
mask_path = main_img_path + dictsettings[0]['train_mask_path']

test_img_path = main_img_path + dictsettings[0]['test_img_path']
test_mask_path = main_img_path + dictsettings[0]['test_mask_path'] 

#%% Training

train_files = sorted(os.listdir(img_path))
train_mask_files = sorted(os.listdir(mask_path))

train_img_df = pd.DataFrame(train_files, columns=['filename'])
train_mask_df = pd.DataFrame(train_mask_files, columns=['filename'])

# Sort images
train_img_df = train_img_df.sample(frac=1,random_state=1).reset_index(drop=True)
train_mask_df = train_mask_df.sample(frac=1,random_state=1).reset_index(drop=True)

poolsize = np.int16(np.shape(train_files)[0]*train_perc)

# Base train dataset
BaseTrain_img = train_img_df[:poolsize]
BaseTrain_mask = train_mask_df[:poolsize]


[train_image_generator, 
 train_mask_generator,
 val_image_generator, 
 val_mask_generator] = ImGenTrain(img_path,
                                  mask_path,
                                  BaseTrain_img,
                                  BaseTrain_mask,
                                  validation_split=validation_split,
                                  DataAugm=1)


train_generator = zip(train_image_generator, train_mask_generator)
valid_generator = zip(val_image_generator, val_mask_generator)

steps_train = train_image_generator.n//train_image_generator.batch_size
steps_valid = val_image_generator.n//val_image_generator.batch_size

#%%
"""
for _ in range(1,2):
    img = train_image_generator.next()
    mask = train_mask_generator.next()
    
    print(img.shape)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img[0])
    plt.subplot(1,2,2)
    plt.imshow(mask[0])
    plt.axis('off')
    plt.show()
"""
#%%

sm.set_framework('tf.keras')
sm.framework()

def TrainModel(train_generator,steps_train,valid_generator,steps_valid,load_weights):

    model = createmodel()
    
    if load_weights==1:
        print('Loading Weigths')
        model.load_weights(ckptmodel_path)
        
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
    
    model.compile('Adam',loss=[sm.losses.JaccardLoss()],
                  metrics=[metrics.iou_score],)
    
    
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


#%% PRE-TRAIN MODEL

model = TrainModel(train_generator,
                   steps_train,
                   valid_generator,
                   steps_valid,
                   load_weights=0)


#%%

def TestingPool(img_path,mask_path,files,maskfiles,print_metrics_summary,path_model):
    
    sm.set_framework('tf.keras')
    sm.framework()

    # BACKBONE = backbone_cnn
    # model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    
    # model = createmodel()
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

    IoU = np.array(IoUmetric)
    meanIoU = np.mean(IoUmetric)
    stdIoU = np.std(IoUmetric)
    
    dicescore = np.array(Dicemetric)
    meandice = np.mean(dicescore)
    stddice = np.std(dicescore)
        
    if print_metrics_summary==True:

        print('------------------------')
        print('Mean IoU: '+str(meanIoU))
        print('Std IoU: '+str(stdIoU))
        print('------------------------')
        
        print('------------------------')
        print('Mean Dice: '+str(meandice))
        print('Std Dice: '+str(stddice))
        print('------------------------')
    
    return Dicemetric,meandice,stddice,meanIoU,stdIoU

#%% TESTING PRE-TRAINED MODEL

test_files = sorted(os.listdir(test_img_path))
test_mask_files = sorted(os.listdir(test_mask_path))

test_img_df = pd.DataFrame(test_files, columns=['filename'])
test_mask_df = pd.DataFrame(test_mask_files, columns=['filename'])

test_image_generator, test_mask_generator=ImGenTest(test_img_path,
                                                       test_mask_path,
                                                       test_img_df,
                                                       test_mask_df,
                                                       DataAugm=0)

path_model = ckptmodel_path
model = createmodel()

Testing_img_list = test_img_df.values.tolist()
Testing_mask_list = test_mask_df.values.tolist()

TestingPool(test_img_path,
            test_mask_path,
            Testing_img_list,
            Testing_mask_list,
            True, 
            path_model)
                       
#%% Data Partition

MeanDiceList = []
StdDiceList = []

BaseTrain_img = train_img_df[:poolsize]
BaseTrain_mask = train_mask_df[:poolsize]

BaseTrain_img_list = train_img_df[:poolsize].values.tolist()
BaseTrain_mask_list = train_mask_df[:poolsize].values.tolist()

Unlabeled_img = train_img_df[poolsize:]
Unlabeled_mask = train_mask_df[poolsize:]

#Query pool size
querypoolsize = np.int16(np.shape(Unlabeled_img)[0]*query_perc)

# Number of iterations
iterations = np.shape(Unlabeled_img)[0]//querypoolsize

# Convert Unlabeled img DF to list
Unlabeled_img_list = Unlabeled_img.values.tolist()
Unlabeled_mask_list = Unlabeled_mask.values.tolist()

#%%
# iterations = 1
for i in range(iterations):
    
    (DiceUnlabeledPool,_,_,_,_) = TestingPool(img_path,
                                              mask_path,
                                              Unlabeled_img_list,
                                              Unlabeled_mask_list,
                                              False,
                                              ckptmodel_path,
                                              
                                              )
    
    new_list = pd.DataFrame(
        {'filename': Unlabeled_img_list,
         'mask_filename': Unlabeled_mask_list,
         'DiceScore': DiceUnlabeledPool
        })
    
    # Sort pool 
    new_list_sort=new_list.sort_values(by=['DiceScore'])
    print('Head')
    print(new_list_sort.head(5))
    print('Tail')
    print(new_list_sort.tail(5))
    
    
    sorted_img_pool = new_list_sort['filename'].values.tolist()
    sorted_mask_pool = new_list_sort['mask_filename'].values.tolist()
    
    query_img_pool = sorted_img_pool[:querypoolsize]
    query_mask_pool = sorted_mask_pool[:querypoolsize]
    
    new_unlabeled_img_pool = sorted_img_pool[querypoolsize:]
    new_unlabeled_mask_pool = sorted_mask_pool[querypoolsize:]
    
    NewTrain_img = BaseTrain_img_list + query_img_pool
    NewTrain_mask = BaseTrain_mask_list + query_mask_pool
    
    # Updated list
    Unlabeled_img_list = new_unlabeled_img_pool
    Unlabeled_mask_list =  new_unlabeled_mask_pool
    
    
    train_img_pool = pd.DataFrame(NewTrain_img, columns=['filename'])
    train_mask_pool = pd.DataFrame(NewTrain_mask, columns=['filename'])    
        
    
    [train_image_generator, 
     train_mask_generator,
     val_image_generator, 
     val_mask_generator] = ImGenTrain(img_path,
                                      mask_path,
                                      train_img_pool,
                                      train_mask_pool,
                                      validation_split=validation_split,
                                      DataAugm=1)
    
    print('************************')
    print('Tamaño de la nueva lista')
    print(len(Unlabeled_img_list))
    print('************************')
    
    train_generator = zip(train_image_generator, train_mask_generator)
    valid_generator = zip(val_image_generator, val_mask_generator)
    steps_train = train_image_generator.n//train_image_generator.batch_size
    steps_val = val_image_generator.n//val_image_generator.batch_size
    
    
    model = createmodel()
    
    # model = TrainModel(train_generator,steps_train,
    #                    load_weights=1)
    
    
    model = TrainModel(train_generator,
                       steps_train,
                       valid_generator,
                       steps_valid,
                       load_weights=1)
        
    (_,meandice,stddice,meanIoU,stdIoU) = TestingPool(test_img_path,
                                                      test_mask_path,
                                                      Testing_img_list,
                                                      Testing_mask_list,
                                                      False,
                                                      ckptmodel_path)
                           
    
    MeanDiceList.append(meandice)
    StdDiceList.append(stddice)
    
    print('**********************************')
    print('Current Mean Dice: '+ str(meandice))
    print('Max Dice: '+ str(np.max(MeanDiceList)))
    print('**********************************')
    
    if meandice>=np.max(MeanDiceList):
        print('Guardando Modelo con Mejor Dice')
        model.save_weights(bestmodel_path)


print('Mean Dice List')

print(MeanDiceList)

print('Std Dice List')

print(StdDiceList)

#%% TESTING MODEL ON TEST-DATASET

# path_model = "C:/Users/Andres/Desktop/BestAL_PC_14122022.h5"
path_model = bestmodel_path

model = createmodel()
model.load_weights(path_model)

Testing_img_list = test_img_df.values.tolist()
Testing_mask_list = test_mask_df.values.tolist()

TestingPool(test_img_path,
            test_mask_path,
            Testing_img_list,
            Testing_mask_list,
            True, 
            path_model)

#%%
# =============================================================================
# Active Learning - Performance report
# =============================================================================


import matplotlib.pyplot as plt
import numpy as np

mean_1 = np.array(MeanDiceList)
std_1 = np.array(StdDiceList)

x = np.arange(len(mean_1))
plt.plot(x, mean_1, 'b-', label='mean_1')
plt.xticks(x)
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.xlabel('Iterations')
plt.ylabel('Dice Score')
# plt.plot(x, mean_2, 'r-', label='mean_2')
# plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
#Activation(activation, kwargs)plt.legend()
plt.show()

#%%

"""
***Evaluation***

"""

import segmentation_models as sm
# from segmentation_models import metrics
sm.set_framework('tf.keras')
sm.framework()

path_model = "C:/Users/Andres/Desktop/BestAL_MV_Exp0_aiffpe.h5"
path_model = bestmodel_path

model = createmodel()
model.load_weights(path_model)

# model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
#%%

path = 'C:/Users/Andres/Documents/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_aiffpe/Testing/MV/'
maskpath = 'C:/Users/Andres/Documents/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_aiffpe/Testing/MV_SG/'


dicescore = []
IoUmetric = []
Dicemetric = []


imsize = 896
scale = 4
scaleimsize = imsize//scale

files = sorted(os.listdir(path))
maskfiles = sorted(os.listdir(maskpath))
 
for imfile,maskfile in zip(files,maskfiles):
# for imfile,maskfile in zip(files[14:15],maskfiles[14:15]):

    imarray = cv.imread(path+imfile)
    imarray = imarray/255                   #Normalize [0 to 1]
    
    maskarray=cv.imread(maskpath+maskfile) 
    
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
    
    # Choose Lung Infection mask (Prediction and groundtruth)
    predmask = np.int16(predmask[:,:,0]>0.5) # Predicted mask
    truemask = np.int16(mask_resized[:,:,2]//255) # Groundtruth mask
    
    # Flattening mask
    predmask_flat = predmask.flatten()
    truemask_flat = truemask.flatten()
    
    # Intersection and Union
    intersection = predmask_flat & truemask_flat
    union = predmask_flat | truemask_flat
    
    if np.sum(truemask_flat)>0:
        
        print(imfile)
        delta=0.0001
        IoU =np.sum(intersection)/(np.sum(union)+delta)
        Dice = 2*np.sum(intersection)/(np.sum(predmask_flat)+np.sum(truemask_flat)+delta)

    Dicemetric.append(Dice)
    IoUmetric.append(IoU)
    
    
    plt.show()
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

print(np.shape(Dicemetric))

print('------------------------')
print('Mean IoU: '+str(meanIoU))
print('Std IoU: '+str(stdIoU))
print('------------------------')

print('------------------------')
print('Mean Dice: '+str(meandice))
print('Std Dice: '+str(stddice))
print('------------------------')
    
#%%

# import cv2 as cv

# a = np.zeros((224,224,3))
# for i in range(3):
#     a[:,:,i]=truemask*255

# cv.imwrite("C:/Users/Andres/Desktop/groundtruth.png", a)

# for i in range(3):
#     a[:,:,i]=predmask*255

# cv.imwrite("C:/Users/Andres/Desktop/prediction.png", a)



#%%

# """