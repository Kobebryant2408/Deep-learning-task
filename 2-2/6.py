# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:58:42 2018

@author: kobe24

主要内容:优化算法(minibatch梯度下降,momentum,adam)
"""

#Optimization Methods
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#1 - Gradient Descent
def update_parameters_with_gd(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(0,L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
        
    return parameters

#2 - Mini-Batch Gradient descent
def random_mini_batches(X,Y,mini_batch_size=64,seed=1):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

#3 - Momentum
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)] + (1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)] + (1-beta)*grads["db"+str(l+1)]
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v["db"+str(l+1)]
        
    return parameters,v

#4 - Adam
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v,s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        
        v_corrected["dW"+str(l+1)] = v["dW"+str(l+1)]/(1-beta1**t)
        v_corrected["db"+str(l+1)] = v["db"+str(l+1)]/(1-beta1**t)
        
        s["dW"+str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*(grads["dW"+str(l+1)]**2)
        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*(grads["db"+str(l+1)]**2)
        
        s_corrected["dW"+str(l+1)] = s["dW"+str(l+1)]/(1-beta2**t)
        s_corrected["db"+str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*(v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon))
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*(v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon))
        
    return parameters,v,s

#5 - Model with different optimization algorithms
train_X,train_Y = load_dataset()
def model(X,Y,layers_dims,optimizer,learning_rate=0.0007,mini_batch_size=64,beta=0.9,
          beta1=0.9,beta2=0.999,epsilon=1e-8,num_epochs=10000,print_cost=True):
    #L = len(layers_dims)
    costs = []
    seed = 10
    t = 0
    
    parameters = initialize_parameters(layers_dims)
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v,s = initialize_adam(parameters)
        
    for i in range(0,num_epochs):
        seed = seed + 1
        mini_batches = random_mini_batches(X,Y,mini_batch_size,seed)
        
        for mini_batch in mini_batches:
            minibatch_X,minibatch_Y = mini_batch
            
            a3,cache = forward_propagation(minibatch_X,parameters)
            
            cost = compute_cost(a3,minibatch_Y)
            
            grads = backward_propagation(minibatch_X,minibatch_Y,cache)
            
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer == "momentum":
                parameters,v = update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t = t+1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
            
        if print_cost and i % 1000 == 0:
            print("Cost after epoch {}:{}".format(i,cost))
            costs.append(cost)
    
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("epochs(per 100)")
    plt.title("Learning rate = "+str(learning_rate))
    plt.show()
    
    return parameters