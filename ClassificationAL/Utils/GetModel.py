# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:02:26 2023

@author: Andres
"""

from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.resnet import ResNet152, ResNet50
from tensorflow.keras.applications import EfficientNetB0 

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from vit_keras import vit

def GetModel():
    
    # model = EfficientNetB0(include_top=True,
    #                         weights=None,
    #                         input_tensor=None,
    #                         input_shape=(224,224,3),
    #                         pooling=None,
    #                         classes=2,
    #                         classifier_activation="softmax")
    
    vit_model = vit.vit_b16(
            image_size = 224,
            activation = 'softmax',
            pretrained = 'C:/Users/Andres/Downloads/ViT-B_16_imagenet21k+imagenet2012.npz',
            include_top = False,
            pretrained_top = False,
            classes = 2)
    
    model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
            tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
            tf.keras.layers.Dense(2, 'softmax')
        ],
        name = 'vision_transformer')
        

    
    return model




# image_size = 224
# batch_size = 16
# n_classes = 3
# EPOCHS = 30



# class Patches(L.Layer):
#     def __init__(self, patch_size):
#         super(Patches, self).__init__()
#         self.patch_size = patch_size

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(
#             images = images,
#             sizes = [1, self.patch_size, self.patch_size, 1],
#             strides = [1, self.patch_size, self.patch_size, 1],
#             rates = [1, 1, 1, 1],
#             padding = 'VALID',
#         )
#         patch_dims = patches.shape[-1]
#         patches = tf.reshape(patches, [batch_size, -1, patch_dims])
#         return patches
    



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
