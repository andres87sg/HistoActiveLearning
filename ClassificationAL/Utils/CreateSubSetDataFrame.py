import pandas as pd
import numpy as np
import os


class CreateSubSetDataFrame:
    def __init__(self, img_path, classes):
        self.img_path = img_path
        self.classes = classes

    def GetSimpleDataFrame(self,img_path,class1,label):
        
        path = img_path + class1 +'/'
        filesfromclass = sorted(os.listdir(path))
        class1 = [class1]*np.shape(filesfromclass)[0]
        pathfromclass = [path]*np.shape(filesfromclass)[0]
        pathfilename = []
        
        for i in range(0,np.shape(filesfromclass)[0]):
            pathfilename.append(pathfromclass[i] + filesfromclass[i])
            
        data = {
                'filename': filesfromclass,
                'pathfile' : pathfilename,                
                'class': class1,
                'label': label}
        
        # self.filesdataframe = pd.DataFrame(data,
                                     # columns=['filename','class','label'])
        return pd.DataFrame(data,columns=['filename','pathfile','class','label'])
    
    def GetMergedDataFrames(self):
        mergedf=pd.DataFrame([],columns=['filename','pathfile','class','label'])
        # for currentclass,i in zip(self.classes,range(2)):
        for currentclass,i in zip(self.classes,range(6)):
            mergedf_aux = self.GetSimpleDataFrame(self.img_path,currentclass,i)
            mergedf = pd.concat([mergedf,mergedf_aux],axis=0)
            
        # self.mergedf = mergedf.reset_index(drop=True)
        return mergedf.reset_index(drop=True)

#%%
"""
imsize = 224
scale = 1
scaleimsize = imsize//scale
batch_size = 2

train_model_epochs = 3

# model_path = '/mnt/rstor/CSE_BME_CCIPD/home/asg143/modelsPC_AL/'
model_path = 'C:/Users/Andres/Desktop/'

ckptmodel_path = os.path.join(model_path,'TempModelCTvsNE.h5')
bestmodel_path = os.path.join(model_path,'BestAL_CTvsNE.h5')

#%%    

# Query pool size (percentage)
query_perc = 0.15

# Initial Training size percentage
train_perc = 0.25

main_path = 'C:/Users/Andres/Desktop/BCMicro/'

# Training Folder Path
trainpath = os.path.join(main_path, "Train/")

# Testing Folder Path
testpath = os.path.join(main_path, "Test/")


#%%
classes = ['MET','NOR']
TrainData = CreateSubSetDataFrame(trainpath,classes).GetMergedDataFrames()
# TestData = CreateSubSetDataFrame(testpath,classes).GetMergedDataFrames()

# TrainData = TrainData.sample(frac=1,random_state=1).reset_index(drop=True)

# train_perc = 0.25
# poolsize = np.int16(np.shape(TrainData)[0]*train_perc)

# BaseTrainData = TrainData[:poolsize]
"""