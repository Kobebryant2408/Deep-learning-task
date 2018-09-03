# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:44:06 2018

@author: kobe24
"""

import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
 
 
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
 
def load_dataset():
    train_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
 
    test_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
     
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
     
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


