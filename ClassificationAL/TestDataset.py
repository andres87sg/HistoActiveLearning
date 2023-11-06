# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:35:12 2023

@author: Andres
"""

import math
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd


# import tensorflow as tf

import tensorflow.keras as keras

# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator

# Custom functions
# from createmodel2 import createmodel

#from preprocessingdata import transform_rgb2lab
# from Testingtools import Testingmodel, PredictUnlabeledDataPool
# from Trainingtools import GeneratorData,ModelTraining,ModelTraining2
from Trainingtools import TestGeneratorData
from CreateSubSetDataFrame import CreateSubSetDataFrame

from sklearn.metrics import confusion_matrix, roc_auc_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#%%

def Testingmodel(TestData,TestImgGenerator,model,path_model):

    # model = GetModel()
    
    # model.load_weights(path_model)

    StepsTest = TestImgGenerator.n//TestImgGenerator.batch_size

    prediction = model.predict(TestImgGenerator,steps = StepsTest+1,verbose=0)
    
    #truelabel = np.array(TestData['label'].astype(float))
    truelabel=TestImgGenerator.classes
    predictedlabel = np.round(prediction[:,0])
    predictedlabel = np.int16(predictedlabel)
    
    tn, fp, fn, tp = confusion_matrix(truelabel,predictedlabel).ravel()
    
    acc = (tp+tn)/( tn + fp + fn + tp)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    auc = roc_auc_score(truelabel,prediction[:,0])
    
    print(f"Accuracy: {acc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"AUC: {auc}")
    
    filename = list(TestData['filename'])
    truelabel = list(TestData['label'])
    predictedlabel = list(predictedlabel)
    
    d = {'filename': filename, 'true': truelabel, 'prediction': predictedlabel}
    dataresults = pd.DataFrame(data=d)
    
    ll = dataresults[(dataresults["true"]==0) & (dataresults["prediction"]==1)]
    a=0
    # class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
    
    return acc, sens, spec, auc

#%%


imsize = 224
scale = 1
scaleimsize = imsize//scale
batch_size = 16
target_size=(224,224)
epochs = 50
# main_path = 'D:/GBM_Project/Current_Experiments/CT_NE_Patches/'
main_path = 'C:/Users/Andres/Desktop/PatchExtractionCT/'
testpath = os.path.join(main_path, "Testing2/")

classes = ['NE','CT']
# TrainData = CreateSubSetDataFrame(trainpath,classes).GetMergedDataFrames()
# TrainData = TrainData.sample(frac=1,random_state=1).reset_index(drop=True)

TestData = CreateSubSetDataFrame(testpath,classes).GetMergedDataFrames()

TestSetImgGenerator,StepsTest = TestGeneratorData(TestData,
                                classes,
                                target_size,
                                batch_size,
                                data_augmentation = False,
                                shuffle = False)



model_path = 'D:/GBM_Project/Experiments/CurrentModels/ModelCTvsNE.h5'
# model_path = 'C:/Users/Andres/Desktop/'

ckptmodel_path = os.path.join(model_path,'TempModelCTvsNE.h5')
bestmodel_path = os.path.join(model_path,'BestAL_CTvsNE.h5')
#%%
# model = GetModel()
from tensorflow.keras.applications import EfficientNetB0 

model = keras.applications.EfficientNetB0(include_top=True,
                        weights=None,
                        input_tensor=None,
                        input_shape=(224,224,3),
                        pooling=None,
                        classes=2,
                        classifier_activation="softmax")

#%%
# model_path = bestmodel_path


model_path = 'D:/GBM_Project/Experiments/CurrentModels/ModelCTvsNE.h5'

model.load_weights(model_path) 


#%%
print(' ------------')
print("| Final test |")
print(' ------------')
acc,sens,spec,auc = Testingmodel(TestData,
                                 TestSetImgGenerator,
                                 model,
                                 model_path)  

#%%




