# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 19:03:21 2018

@author: kobe24

注意问题:5.py,5.1.py,5.2.py是一体的,分别是初始化,正则化,梯度检测;
        其中5.py与init_utils.py;5.1.py与reg_utils.py,testCases.py;5.2.py与testCases2.py,gc_utils.py分别为一体的
"""

###############################################################################
#Part 1：Initialization
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid,relu,compute_loss,forward_propagation,backward_propagation
from init_utils import update_parameters,predict,load_dataset,plot_decision_boundary,predict_dec

plt.rcParams['figure.figsize'] = (7.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

#1 - Neural Network model
def model(X,Y,learning_rate = 0.01,num_iterations = 15000,print_cost = True,initialization = "he"):
    #grads = {}
    costs = []
    #m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]
    
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
        
    for  i in range(0,num_iterations):
        a3, cache = forward_propagation(X,parameters)
        cost = compute_loss(a3,Y)
        grads = backward_propagation(X,Y,cache)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i,cost))
            costs.append(cost)
    
    #绘制学习曲线        
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate =' + str(learning_rate))
    
    return parameters

#2 - Zero initialization
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
        
        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l],1))
        
    return parameters

#3 - Random initialization(large value:W的初始化后面有个乘10)
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
        
        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l],1))
    
    return parameters

#4 - He initialization
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
        
        assert(parameters["W"+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layers_dims[l],1))
    
    return parameters

