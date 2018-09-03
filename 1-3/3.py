# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:39:17 2018

@author: kobe24

注意问题:第四部分定义的函数都是基于第二部分的数据集,
        若用第五部分的数据集,注意输入数据集的下标
        
构建单隐层神经网络的步骤(涉及的所有函数):50-140(辅助函数),143-161(模型)

3.py,testCases_v2.py,planar_utils.py是一体的
"""

#Planar data classification with one hidden layer

#1 - Packages
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

#2 - Dataset
X,Y = load_planar_dataset()

#3 - Simple Logistic Regression
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

#plot_decision_boundary(lambda x:clf.predict(x),X,Y)#绘制逻辑回归决策边界
#plt.title('Logistic Regression')

#4 - Neural Network model

#Reminder: The general methodology to build a Neural Network is to: 
#1. Define the neural network structure ( # of input units, # of hidden units, etc). 
#2. Initialize the model’s parameters 
#3. Loop: 
#- Implement forward propagation 
#- Compute loss 
#- Implement backward propagation to get the gradients 
#- Update parameters (gradient descent)

#You often build helper functions to compute steps 1-3 and then merge them into one function we call nn_model(). Once you’ve built nn_model() and learnt the right parameters, you can make predictions on new data.

#4.1 - Defining the neural network structure
def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)

#4.2 - Initialize the model’s parameters
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)         
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"W1":W1,
                  "b1":b2,
                  "W2":W2,
                  "b2":b2}
    return parameters

#4.3 - The Loop
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)    
    
    assert(A2.shape == (1,X.shape[1]))
    
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}
    return A2,cache
def compute_cost(A2,Y,parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = -(1.0/m)*np.sum(logprobs)
    
    cost = np.squeeze(cost)
    assert(isinstance(cost,float))
    
    return cost
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    #W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = 1.0/m*np.dot(dZ2,A1.T)
    db2 = 1.0/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = 1.0/m*np.dot(dZ1,X.T)
    db1 = 1.0/m*np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {"dW1":dW1,
             "db1":db1,
             "dW2":dW2,
             "db2":db2}
    return grads
def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    return parameters

#4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model()
def nn_model(X,Y,n_h,num_iterations = 10000,print_cost=True):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
    
        grads = backward_propagation(parameters,cache,X,Y)
    
        parameters = update_parameters(parameters,grads,learning_rate=1.2)
        
        if print_cost and i%1000 == 0:
            print("Cost after iteration %i:%f"%(i,cost))
            
    return parameters

#4.5 Predictions
def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters)
    predictions = (A2 > 0.5)
    
    return predictions

#4.6 - Tuning hidden layer size (optional/ungraded exercise)

def figure():
    plt.figure(figsize=(16,32))
    hidden_layer_sizes = [1,2,3,4,5,20,50]
    for i,n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5,2,i+1)
        plt.title("Hidden Layer of size %d" %n_h)
        parameters = nn_model(X,Y,n_h,num_iterations=5000)
        plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
        
        predictions = predict(parameters,X)
        accuracy = float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print("Accuracy for {} hidden units: {}%".format(n_h,accuracy))
        
#5 - Performance on other datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X1, Y1 = datasets[dataset]
X1, Y1 = X1.T, Y1.reshape(1, Y1.shape[0])

# make blobs binary
if dataset == "blobs":
    Y1 = Y1%2

# Visualize the data
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);