# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:14:37 2018

@author: kobe24

遇到的问题:part2中第二部分h5py文件打不开,对应load_data()函数(在dnn_app_utils_v2.py中)
特别注意:part2的第二部分是数据预处理(将图片数据降维),包括加载,重构,标准化(归一化)数据集

构建L层神经网络的步骤(涉及的所有函数):51-172(辅助函数),221-244(模型)

4.py,dnn_app_utils_v2.py,testCases_v3.py,dnn_utils_v2.py是一体的
"""

###############################################################################
#Part 1：Building your Deep Neural Network: Step by Step

#1 - Packages
import numpy as np
#import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#2 - Outline of the Assignment

#3 - Initialization
#3.1 - 2-layer Neural Network
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    
    return parameters

#3.2 - L-layer Neural Network（L包括了输入层）
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layer_dims[l],1))
    
    return parameters

#4 - Forward propagation module
#4.1 - Linear Forward
def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    return Z,cache

#4.2 - Linear-Activation Forward
def linear_activation_forward(A_prev,W,b,activation):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
        
    assert(A.shape == (W.shape[0],A.shape[1]))
    cache = (linear_cache,activation_cache)
    
    return A,cache

#4.3 - L-Layer Model forward_propagration
def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
         
    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches

#5 - Cost function
def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = -(1/m)*np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost    

#6 - Backward propagation module
#6.1 - Linear backward
def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev,dW,db

#6.2 - Linear-Activation backward
def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    return dA_prev,dW,db

#6.3 - L-Model Backward_propagation
def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    #m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    
    current_cache = caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
        
    return grads

#6.4 - Update Parameters
def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] = learning_rate*parameters["W"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] = learning_rate*parameters["b"+str(l+1)]
    return parameters

###############################################################################
#Part 2：Deep Neural Network for Image Classification: Application

#1 - Packages
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#2 - Dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#3 - Architecture of your model
#3.1 - 2-layer neural network

#3.2 - L-layer deep neural network

#3.3 - General methodology
#As usual you will follow the Deep Learning methodology to build the model: 
#1. Initialize parameters / Define hyperparameters 
#2. Loop for num_iterations: 
#a. Forward propagation 
#b. Compute cost function 
#c. Backward propagation 
#d. Update parameters (using parameters, and grads from backprop) 
#4. Use trained parameters to predict labels\

#4 - Two-layer neural network(该模型已经在作业3写过了)

#5 - L-layer Neural Network
layers_dims = [12288,20,7,5,1]
def L_layer_model(X,Y,layers_dims,learning_rate,num_iterations = 3000, print_cost = True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0,num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration %i:%f"%(i,cost))
        #if print_cost and i%100 == 0:
            costs.append(cost)
    
    plt.plot(np.squeeze(cost))
    plt.ylabel('cost')
    plt.xlabel('iterations(per 100)')
    plt.title('Learning rate = '+ str(learning_rate))
    plt.show()
    
    return parameters

#6) Results Analysis
    
#7) Test with your own image (optional/ungraded exercise)