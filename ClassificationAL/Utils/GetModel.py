# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:02:26 2023

@author: Andres
"""

from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.resnet import ResNet152, ResNet50
from tensorflow.keras.applications import EfficientNetB0 


def GetModel():
    
    model = EfficientNetB0(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=(224,224,3),
                            pooling=None,
                            classes=2,
                            classifier_activation="softmax")
    
    return model


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
