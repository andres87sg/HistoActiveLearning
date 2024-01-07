# =============================================================================
# 
# Reference
# https://github.com/qubvel/segmentation_models
# =============================================================================

import json
# import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

# Segmentation_models library
# import segmentation_models as sm
from Utils.GetModel3 import GetModel
# from Utils.GetModel2 import GetModel
# from Utils.GetModel3 import GetModel
from Utils.GetTrainModel import TrainModel
from Utils.GetTestModel import TestPool
from Utils.GetImgSubsetGenerator import ImGenTrain, ImGenTest

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open('input_variables.json','r') as f:
   data = json.load(f)
   
dictsettings = data["settings"]
main_img_path = dictsettings[0]['img_path']

ModelPath =  dictsettings[1]['ModelPath']
BestModelName = dictsettings[1]['BestModelName'] 
TempModelName = dictsettings[1]['TempModelName']

experiment_name = dictsettings[0]['experiment_name']

# Parameters

imsize = dictsettings[2]["imsize"]
scale = dictsettings[2]["scale"]
batch_size = dictsettings[2]["batch_size"]
train_model_epochs = dictsettings[2]["train_model_epochs"]
scaleimsize = imsize//scale

ckptmodel_path = ModelPath + TempModelName
bestmodel_path = ModelPath + BestModelName

# Verbose -> Model Training Tracking

verbosemodelcheckpoint = dictsettings[4]["verbosemodelcheckpoint"]
verbosemodelfit = dictsettings[4]["verbosemodelfit"]
verboseearlystopping =  dictsettings[4]["verboseearlystopping"]
earlystoppingepochs =  dictsettings[4]["earlystoppingepochs"]
    
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

print('************************')
print('*** ' + experiment_name + ' ***')
print('************************')
print('------ Parameters -------')
print(f'Epochs: {train_model_epochs}')
print(f'Batch size: {batch_size}')
print(f'Image size: {imsize} pix scaled {scale} times ({scaleimsize} pix)')
print(f'Base train percentage: {train_perc*100:.0f}% ')
print('|')
print(f'|--> Split: Training {(1-validation_split)*100:.0f}% and Validation {(validation_split)*100:.0f}%')
print(f'Query percentage (training set): {query_perc*100:.0f}%')
print('____________________')

## Training

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

#%%
[train_image_generator, train_mask_generator,
 val_image_generator, val_mask_generator] = ImGenTrain(img_path,
                                  mask_path,
                                  BaseTrain_img,
                                  BaseTrain_mask,
                                  validation_split=validation_split,
                                  DataAugm=1)


train_generator = zip(train_image_generator, train_mask_generator)
valid_generator = zip(val_image_generator, val_mask_generator)

steps_train = np.int8(np.ceil(train_image_generator.n/train_image_generator.batch_size))
steps_valid = np.int8(np.ceil(val_image_generator.n//val_image_generator.batch_size))

print("------ Training and validation split -------")
print(f'Base training and validation samples: {train_image_generator.n+val_image_generator.n}')
print(f'Training samples: {train_image_generator.n} -> {train_image_generator.n/(train_image_generator.n+val_image_generator.n)*100:.1f}%')
print(f'Validation samples: {val_image_generator.n} -> {val_image_generator.n/(train_image_generator.n+val_image_generator.n)*100:.1f}%')
print("--------------------------------------------")
#%%
show_image_samples = False

if show_image_samples:
    for _ in range(1,4):
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

#%% PRE-TRAIN MODEL

model = TrainModel(train_generator,
                   steps_train,
                   valid_generator,
                   steps_valid,
                   load_weights=0, 
                   model_path=ckptmodel_path)

# load_weigts: 0) from scratch 1) Pretrained 2) Imagenet pretrained

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

print('----> Testing Base Model <----')

path_model = ckptmodel_path
model = GetModel()

#%%
Testing_img_list = test_img_df.values.tolist()
Testing_mask_list = test_mask_df.values.tolist()

(DiceUnlabeledPool,_,_,_,_)  = TestPool(test_img_path,
                                test_mask_path,
                                Testing_img_list,
                                Testing_mask_list,
                                True, 
                                path_model,
                                model)
                                           
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

print('_________________________')
print(f"Initial Unlabeled Pool Size: {len(Unlabeled_img_list)}")
print('_________________________')

#%%+
print('*****************')
print('Active Learning')
print('*****************')

import random
pd.options.display.max_columns = None

# iterations = 1
for iteration in range(iterations-1):

    print(f"Iteration Numbrer {iteration +1} of {iterations-1}")
    
    print("Testing Unlabeled Pool")
    (DiceUnlabeledPool,_,_,_,_) = TestPool(img_path,
                                              mask_path,
                                              Unlabeled_img_list,
                                              Unlabeled_mask_list,
                                              False,
                                              ckptmodel_path,
                                              model)
    
    random.seed(1)
    randomlist = random.sample(range(0, len(Unlabeled_img_list)), 
                               len(Unlabeled_img_list))
    
    new_list = pd.DataFrame(
        { 'index': randomlist,
          'filename': Unlabeled_img_list,
          'mask_filename': Unlabeled_mask_list,
         'DiceScore': DiceUnlabeledPool
        })
    
    new_list.drop(columns={'filename'})
    
    sort_by = 'DiceScore'

    if sort_by == 'DiceScore':
    
            
        new_list_sort=new_list.sort_values(by=['DiceScore'])
        print('Head')
        print(new_list_sort.head(3))
        # print(new_list_sort[{'filename','DiceScore'}].head(3))
        print('Tail')
        print(new_list_sort.tail(3))
        # print(new_list_sort[{'filename','DiceScore'}].tail(3))
    
    if sort_by == 'Random':
        new_list_sort=new_list.sort_values(by=['index'])
        print('Head')
        # print(new_list_sort[{'filename','DiceScore'}].head(3))
        print(new_list_sort.head(3))
        
        print('Tail')
        print(new_list_sort.tail(3))
        # print(new_list_sort[{'filename','DiceScore'}].tail(3))
    
    
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
        
    
    [train_image_generator, train_mask_generator,
     val_image_generator, val_mask_generator] = ImGenTrain(img_path,
                                                           mask_path,
                                                           train_img_pool,
                                                           train_mask_pool,
                                                           validation_split=validation_split,
                                                           DataAugm=1)
    
    print('_________________________________')
    print(f"New Unlabeled-Pool Size: {len(Unlabeled_img_list)}")
    print('---------------------------------')
    
    train_generator = zip(train_image_generator, train_mask_generator)
    valid_generator = zip(val_image_generator, val_mask_generator)
    
    steps_train = np.int8(np.ceil(train_image_generator.n/train_image_generator.batch_size))
    steps_valid = np.int8(np.ceil(val_image_generator.n//val_image_generator.batch_size))
    
    model = TrainModel(train_generator,
                       steps_train,
                       valid_generator,
                       steps_valid,
                       load_weights=1,
                       model_path=ckptmodel_path)
    
    print("Testing Model - Unlabeled Pool")
    
    (_,meandice,stddice,meanIoU,stdIoU) = TestPool(test_img_path,
                                                      test_mask_path,
                                                      Testing_img_list,
                                                      Testing_mask_list,
                                                      False,
                                                      ckptmodel_path,
                                                      model)
                           
    
    MeanDiceList.append(meandice)
    StdDiceList.append(stddice)
    
    print('**********************************')
    print(f"Current Dice-Score: {str(meandice)}")
    print(f"Best achieved Dice-Score: {np.max(MeanDiceList)}")
    if meandice>=np.max(MeanDiceList):
        print('----> Saving Model Best Dice-Score in: ')
        print(bestmodel_path)
        model.save_weights(bestmodel_path)
    print('**********************************')
    

#%%
print('******** Summary: Dice-Score *********')

ResultsSummary = pd.DataFrame({'Mean': MeanDiceList,'Std': StdDiceList})
print(ResultsSummary.round(3))

#%% TESTING MODEL ON TEST-DATASET
from Utils.GetTestModel import TestPool

print('----> Testing Best Model <----')

path_model = bestmodel_path

model = GetModel()
model.load_weights(path_model)

Testing_img_list = test_img_df.values.tolist()
Testing_mask_list = test_mask_df.values.tolist()

TestPool(test_img_path,
            test_mask_path,
            Testing_img_list,
            Testing_mask_list,
            True, 
            path_model,
            model)

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


# ***Evaluation***

# import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

# path_model = "D:/GBM_Project/Experiments/CurrentModels/PCModel_EffUNet_01022024_Exp1.h5"
# path_model = "D:/GBM_Project/Experiments/CurrentModels/PCModel_EffUNet_18102023_Exp1.h5"
# path_model = bestmodel_path

model = GetModel()
model.load_weights(path_model)

# model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')



path = 'D:/JournalExperiments/PC/TCGA/PC_1792_ChL/Training/PC/'
maskpath = 'D:/JournalExperiments/PC/TCGA/PC_1792_ChL/Training/PC_SG/'

# path = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_data_augm/Testing/MV/'
# maskpath = 'D:/GBM_Project/Current_Experiments/MV_Patches/MV_896_ChA_data_augm/Testing/MV_SG/'
# 

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
    truemask = np.int16(mask_resized[:,:,0]//255) # Groundtruth mask
    
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
    plt.title(imfile[:-7])
    
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
print(f'IoU Score: {meanIoU:.3f} +- {stdIoU:.3f}')
print('------------------------')
print(f'Dice Score: {meandice:.3f} +- {stddice:.3f}')
print('------------------------')


