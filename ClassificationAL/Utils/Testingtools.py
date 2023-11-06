import numpy as np
#import sklearn as sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from tensorflow.keras.applications import EfficientNetB0 
#from tensorflow.keras.applications.convnext import ConvNeXtTiny
#from tensorflow.keras.applications.resnet import ResNet152, ResNet50
from Utils.GetModel import GetModel


from keras.preprocessing.image import ImageDataGenerator

def Testingmodel(TestData,TestSetImgGenerator,model,path_model,verbose):
    
    model = GetModel()
    model.load_weights(path_model)

    # StepsTest = TestSetImgGenerator.n//TestSetImgGenerator.batch_size
    
    StepsTest = np.int16(np.round(TestSetImgGenerator.n/TestSetImgGenerator.batch_size))+1
    
    prediction = model.predict(TestSetImgGenerator,
                               steps = StepsTest,
                               verbose = verbose)
    
    predictedlabel = np.argmax(prediction,axis=1)
    truelabel=np.array(TestSetImgGenerator.classes)
    
    tn, fp, fn, tp = confusion_matrix(truelabel,predictedlabel).ravel()
    
    acc = (tp+tn)/( tn + fp + fn + tp)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    auc = roc_auc_score(truelabel,prediction[:,1])
    
    print(f"Accuracy: {acc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"AUC: {auc}")
    
    return acc, sens, spec, auc
    

def PredictUnlabeledDataPool(Data,classes,target_size,batch_size,model,path_model):
    
    # path_model = ckptmodel_path
    model = GetModel()
    model.load_weights(path_model)   
    
    datagen = ImageDataGenerator(rescale=1./255,)
    DataGenerator = datagen.flow_from_dataframe(Data,
                                                x_col='pathfile', 
                                                classes=classes,
                                                class_mode='categorical', 
                                                target_size=target_size,  
                                                batch_size=batch_size,
                                                shuffle=False,
                                                seed=1)
    
    # StepsTest = DataGenerator.n//DataGenerator.batch_size
    StepsTest = np.int16(np.round(DataGenerator.n/DataGenerator.batch_size))+1
    
    prediction = model.predict(DataGenerator,
                               steps = StepsTest,
                               verbose=0)
    
    return prediction
    

    
    # model = ConvNeXtTiny(
    #                         model_name='convnext_tiny',
    #                         include_top=True,
    #                         include_preprocessing=True,
    #                         weights=None,
    #                         input_tensor=None,
    #                         input_shape=(224,224,3),
    #                         pooling=None,
    #                         classes=6,
    #                         classifier_activation='softmax')
    
    # model = ResNet152(
    #                         include_top=True,
    #                         weights=None,
    #                         input_tensor=None,
    #                         input_shape=(224,224,3),
    #                         pooling=None,
    #                         classes=6,
    #                         classifier_activation="softmax"
    #                         )
    # model = ResNet50(
    #                         include_top=True,
    #                         weights=None,
    #                         input_tensor=None,
    #                         input_shape=(224,224,3),
    #                         pooling=None,
    #                         classes=6,
    #                         classifier_activation="softmax"
    #                         )
    
    return model




