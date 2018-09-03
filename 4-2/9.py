# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:25:53 2018

@author: kobe24

主要内容:keras教程
主要问题:第五部分,keras的可视化工具安装有问题,包括(from Ipython.display import SVG
                                                与from keras.utils import plot_model)
"""

###############################################################################
#Part 1：Keras tutorial - the Happy House
import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
#from Ipython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#1 - The Happy House
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

#2 - Building a model in Keras
def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(32,(7,7),stride=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2),name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    
    model = Model(inputs = X_input,outputs = X,name='HappyModel')
    
    return model

def HappyModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2),name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    
    model = Model(inputs = X_input,outputs = X,name='HappyModel')
    
    return model

happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer = "Adam",loss = "binary_crossentropy",metrics = ["accuracy"])
happyModel.fit(x = X_train,y = Y_train,epochs = 10,batch_size = 32)
preds = happyModel.evaluate(X_test,Y_test)
print()
print("Loss = "+str(preds[0]))
print("Test Accuracy = "+str(preds[1]))
#4 - Test with your own image (Optional)

#5 - Other useful functions in Keras (Optional)
happyModel.summary()

#plot_model(happyModel, to_file='HappyModel.png')
#SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))