import math
import json
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from Utils.Testingtools import Testingmodel, PredictUnlabeledDataPool
from Utils.Trainingtools import GeneratorData,ModelTraining
from Utils.Trainingtools import TestGeneratorData
from Utils.CreateSubSetDataFrame import CreateSubSetDataFrame
from Utils.GetModel import GetModel

# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



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

ModelPath =  dictsettings[1]['ModelPath']
BestModelName = dictsettings[1]['BestModelName'] 
TempModelName = dictsettings[1]['TempModelName']

imsize = dictsettings[2]["imsize"]
scale = dictsettings[2]["scale"]
batch_size = dictsettings[2]["batch_size"]
train_model_epochs = dictsettings[2]["train_model_epochs"]
scaleimsize = imsize//scale
target_size=(imsize,imsize)

ckptmodel_path = os.path.join(ModelPath,TempModelName)
bestmodel_path = os.path.join(ModelPath,BestModelName)

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

# # Training Folder Path
train_img_path = os.path.join(main_img_path, dictsettings[0]['train_img_path'])

# # Testing Folder Path
test_img_path = os.path.join(main_img_path, dictsettings[0]['test_img_path'])


#%%

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

#%%

classes = ['NE','CT']

TrainData = CreateSubSetDataFrame(train_img_path,classes).GetMergedDataFrames()
TrainData = TrainData.sample(frac=1,random_state=1).reset_index(drop=True)

TestData = CreateSubSetDataFrame(test_img_path,classes).GetMergedDataFrames()

poolsize = np.int16(np.shape(TrainData)[0]*train_perc)

BaseTrainData = TrainData[:poolsize]
#%%

TrainImgGenerator, StepsTrain, ValidImgGenerator, StepsValid = GeneratorData(BaseTrainData,
                                    classes,
                                    target_size,
                                    batch_size,
                                    data_augmentation = True,
                                    shuffle = True,
                                    validation_split = validation_split)

TestSetImgGenerator,StepsTest = TestGeneratorData(TestData,
                                classes,
                                target_size,
                                batch_size,
                                data_augmentation = False,
                                shuffle = False)


#%%

"""
for _ in range(1,3):
    img = TrainImgGenerator.next()
    img2 = img[0][0]
    plt.figure()
    plt.imshow(img2)
    plt.axis('off')
"""
#%%

# Training initial model
model = ModelTraining(TrainImgGenerator,StepsTrain,
                      ValidImgGenerator,StepsValid,
                      train_model_epochs,ckptmodel_path,verbose=verbosemodelfit)

print('saving pretrained model')
model.save_weights(ckptmodel_path)

# Testing pre-trained model
print('*** Testing pre-trained model ***')
acc,sens,spec,auc = Testingmodel(TestData,
                                 TestSetImgGenerator,
                                 model,
                                 ckptmodel_path,
                                 verbose = verbosemodelfit)    

#%%

AccList = []
SensList = []
SpecList = [] 
AUCList = []

#Base Train Data
BaseTrainData = TrainData[:poolsize] 

#Unlabeled pool data
UnlabeledData = TrainData[poolsize:] 

# Query pool size
QueryPoolSize = np.int16(np.shape(UnlabeledData)[0]*query_perc)

# Number of iterations (data/poolsize)
iterations = np.int16(np.round(np.shape(UnlabeledData)[0]/QueryPoolSize))

#%%

def savingbestmodel(acc,AccList,bestmodel_path):
    maxacc = np.max(AccList)
    
    if acc>=maxacc:
        print("Saving Best model")
        model.save_weights(bestmodel_path)
    else:
        print("Model is not saved")


def CalculateEntropySampling(prediction):
    p = prediction[:,0]
    q = 1-p
    delta = 0.000001
    SumEnt = np.abs(-p*np.log(p+delta)/np.log(2)-q*np.log(q+delta)/np.log(2))
    return SumEnt

def CalculateEntropySlopeSampling(prediction):
    p = prediction[:,0]
    q = 1-p
    delta = 0.000001
    SumEnt = np.abs(-p*np.log(p+delta)/np.log(2)-q*np.log(q+delta)/np.log(2))
    return SumEnt


for iteration in range(iterations):
    print(f"Iteration Numbrer {iteration +1} of {iterations}")

    path_model = ckptmodel_path
    prediction = PredictUnlabeledDataPool(UnlabeledData,
                                          classes,
                                          target_size,
                                          batch_size,
                                          model,
                                          path_model)
    
    predictedlabel = np.argmax(prediction,axis=1)
    
    truelabel = np.array(UnlabeledData['label']).astype(float)

    # score = CalculateEntropySampling(prediction)
    
    # Score: Difference between true and prediction
    score = 1-np.abs(truelabel-prediction[:,1])
    
    # Adding score to Unlabeled Data pool
    ScoredUnlabeledData = pd.DataFrame(
        {'filename': UnlabeledData['filename'],
         'pathfile': UnlabeledData['pathfile'],
          'class':   UnlabeledData['class'],
         'label':   UnlabeledData['label'],
         'Prediction': prediction[:,0],
         'Score': np.round(score, 4), 
        }).sort_values(by=['Score'],ascending=False).sort_values(by=['Score'],
                                                              ascending=False)
                                                                 
    ## Printing Unlabeled Data with scores to verify that data is right sorted
    print('Head')
    print(ScoredUnlabeledData.head(3))
    print('Tail')
    print(ScoredUnlabeledData.tail(3))
        
    ## Sorting Unlabeled Data by Prediction Score
    UnlabeledDataSorted = ScoredUnlabeledData[['filename',
                                               'pathfile',
                                               'class',
                                               'label']]
    

    # Updating Unlabeled Data and Query Data pool
    UpdatedUnlabeledData = UnlabeledDataSorted[QueryPoolSize:]
    QueryDataPool = UnlabeledDataSorted[:QueryPoolSize]

    FeedbackData = pd.concat([BaseTrainData,QueryDataPool],axis=0)
    

    print("FeedBack Data")
    FeedBackImgGenerator,StepsTrain, ValidationImgGenerator,StepsValid = GeneratorData(FeedbackData,
                                        classes,
                                        target_size,
                                        batch_size,
                                        data_augmentation = False,
                                        validation_split = validation_split,
                                        shuffle = False)    
 
    model = ModelTraining(FeedBackImgGenerator,
                          StepsTrain,
                          ValidationImgGenerator,
                          StepsValid,
                          train_model_epochs,
                          ckptmodel_path,
                          verbose=verbosemodelfit)
   
    # Updating Unlabeled Data
    UnlabeledData = UpdatedUnlabeledData.copy()
    
    # Testing Model
    path_model = ckptmodel_path
    model.load_weights(path_model)    
    
    print("--- Testing current model ---")
    acc,sens,spec,auc = Testingmodel(TestData,
                                     TestSetImgGenerator,
                                     model,
                                     ckptmodel_path,
                                     verbose=verbosemodelfit)  
    
    AccList.append(np.round(acc,4))
    SensList.append(np.round(sens,4))
    SpecList.append(np.round(spec,4))
    AUCList.append(np.round(auc,4))
    
    print('| -------------------------|')
    print(f" Current Accuracy: {acc} ")
    print(f" Max Accuracy: {np.max(AccList)} ")
    print('|-------------------------|')
    
    savingbestmodel(np.round(acc,4),AccList,bestmodel_path)
    
    
    print('************************')
    print(f"Length of the new list: {len(UnlabeledData)}")
    
#%%

print('Accuracy List')
print(AccList)
print('AUC List')
print(AUCList)


#%%

acc,sens,spec,auc = Testingmodel(TestData,
                                 TestSetImgGenerator,
                                 model,
                                 bestmodel_path,
                                 verbose = verbosemodelfit)  


#%%
model_path = bestmodel_path

model = GetModel()
model_path = 'D:/ISBI Experiments/Models/Classif_31Oct2023_ent.h5'
model.load_weights(model_path) 

classes = ['NE','CT']

test_img_path = 'D:/ISBI Experiments/IvyGap_TCGA/Testing_IVYGAP/' 

TestData = CreateSubSetDataFrame(test_img_path,classes).GetMergedDataFrames()

TestSetImgGenerator,StepsTest = TestGeneratorData(TestData,
                                classes,
                                target_size,
                                batch_size,
                                data_augmentation = False,
                                shuffle = False)

#%%
print(' ------------')
print("| Final test |")
print(' ------------')
acc,sens,spec,auc = Testingmodel(TestData,
                                  TestSetImgGenerator,
                                  model,
                                  model_path,
                                  verbose = 1)  

#%%
