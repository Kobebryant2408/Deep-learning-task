# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:41:33 2018

@author: kobe24

主要内容:是一个有监督的多类(C=6)分类问题

主要问题:明明多层CNN的效果就非常好,9.1.py为什么还要用resnet模型
"""

#1.加载数据
import numpy as np
import h5py

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_dataset():
    train_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
 
    test_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
     
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
     
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#2.利用keras构建模型
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

def model(input_shape):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3,3),name='zeropad')(x_input)
    
    x = Conv2D(32,(3,3),strides=(1,1),name='conv0')(x)
    x = BatchNormalization(axis=3,name='bn0')(x)
    x = Activation('relu',name='acti0')(x)
    x = MaxPooling2D((2,2),name='max_pool0')(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),name='conv1')(x)
    x = BatchNormalization(axis=3,name='bn1')(x)
    x = Activation('relu',name='acti1')(x)
    x = MaxPooling2D((2,2),name='max_pool1')(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),name='conv2')(x)
    x = BatchNormalization(axis=3,name='bn2')(x)
    x = Activation('relu',name='acti2')(x)
    x = MaxPooling2D((2,2),name='max_pool2')(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv3')(x)
    x = BatchNormalization(axis=3,name='bn3')(x)
    x = Activation('relu',name='activ3')(x)
    x = MaxPooling2D((1,1),name='max_pool3')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(6,activation='softmax',name='fc')(x)
    
    model = Model(inputs=x_input,outputs=x,name='model')
    return model

import keras
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses = {'batch':[],'epoch':[]}
        self.accuracy = {'batch':[],'epoch':[]}
        self.val_loss = {'batch':[],'epoch':[]}
        self.val_acc = {'batch':[],'epoch':[]}
    
    def on_batch_end(self,batch,logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self,batch,logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
    def loss_plot(self,loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters,self.accuracy[loss_type],'r',label='train acc')
        plt.plot(iters,self.losses[loss_type],'g',label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters,self.val_acc[loss_type],'b',label='val acc')
            plt.plot(iters,self.val_loss[loss_type],'k',label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()
        
model = model(X_train.shape[1:])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = LossHistory()
model.fit(x=X_train,y=Y_train,batch_size=32,epochs=25,
          verbose=1,
          shuffle=True,
          validation_split=0.3,
          callbacks=[history])
history.loss_plot('epoch')

preds = model.evaluate(x=X_test,y=Y_test)
print('test loss ='+str(preds[0]))
print('test accuracy ='+str(preds[1]))
model.summary()