# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:29:13 2018

@author: kobe24

遇到的问题:h5py文件打不开,对应load_dataset()函数
特别注意:part2的第二部分是数据预处理(将图片数据降维)
               第六部分是分析代价函数得到最优参数(绘制学习曲线)
               
构建逻辑回归模型的步骤(涉及的所有函数):93-156(辅助函数),159-176(模型)
"""

###############################################################################
#Part 1：Python Basics with Numpy (optional assignment)

#1 - Building basic functions with numpy
#1.1- sigmoid function, np.exp()
import math
import numpy as np

def basic_sigmoid(x):
    s = 1.0/(1.0+1/math.exp(x))
    return s

def sigmoid2(x):
    s = 1.0/(1.0+1/np.exp(x))
    return s
#1.2 - Sigmoid gradient
def sigmoid_grad(x):
    s = 1.0/(1.0+1/np.exp(x))
    dx = s*(1-s)
    return dx
#1.3- Reshaping arrays
def image2vector(image):
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2],1))
    return v
    print("image2vector(image) = "+str(image2vector(image)))
#1.4- Normalizing rows
def normlizeRows(x):
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    x = x/x_norm
    return x
#1.5- Broadcasting and the softmax function    
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s

#2 - Vectorization
#2.1 Implement the L1 and L2 loss functions
def L1(yhat,y):
    loss = np.sum(np.abs(y-yhat))
    return loss
def L2(yhat,y):
    loss = np.sum(np.power(yhat,y),2)
    return loss

###############################################################################
#Part 2： Logistic Regression with a Neural Network mindset

#1 - Packages
#import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset

#2 - Overview of the Problem set
def load_dataset():
    train_dataset = h5py.File('E:\\Spyder\\神经网络与深度学习\\Logistic Regression\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
 
    test_dataset = h5py.File('E:\\Spyder\\神经网络与深度学习\\Logistic Regression\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
     
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
     
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#3 - General Architecture of the learning algorithm
    
#4 - Building the parts of our algorithm 
#4.1 - Helper functions
def sigmoid(z):
    s = 1.0/(1+1/np.exp(z))
    return s
#4.2 - Initializing parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w, b    
#4.3 - Forward and Backward propagation
def propagate(w,b,X,Y):
    m = X.shape[1]
    #forward propagation
    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #backward propagation
    dw = (1.0/m)*np.dot(X,(A-Y).T)
    db = (1.0/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw":dw,"db":db}
    
    return grads,cost
#4.4 - Optimization
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 == 0:
            print("Cost dfter iteration %i: %f" %(i,cost))
    
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    
    return params,grads,costs
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

#5 - Merge all functions into a model
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    w,b = initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))
    
    d = {"costs":costs,
         "Y_prediction_train":Y_prediction_train,
        "Y_prediction_test":Y_prediction_test,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations}
    return d

#6 - Further analysis (optional/ungraded exercise)
    
#7 - Test with your own image (optional/ungraded exercise)
