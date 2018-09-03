# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:03:25 2018

@author: kobe24

任务:利用tensorflow写神经网络
特别注意:第二部分为多类分类问题,一开始要加载数据集并进行数据预处理;
        所采用的算法为Adam应用于mini_batch梯度下降(tensorflow实现)
"""

#TensorFlow Tutorial
#1 - Exploring the Tensorflow Library
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset,random_mini_batches,convert_to_one_hot,predict

np.random.seed(1)

#Writing and running programs in TensorFlow has the following steps:

#1.Create Tensors (variables) that are not yet executed/evaluated.
#2.Write operations between those Tensors.
#3.Initialize your Tensors.
#4.Create a Session.
#5.Run the Session. This will run the operations you’d written above.


#1.1 - Linear function
def linear_function():
    np.random.seed(1)
    
    X = tf.constant(np.random.randn(3,1),name = 'X')
    W = tf.constant(np.random.randn(4,3),name = 'W')
    b = tf.constant(np.random.randn(4,1),name = 'b')
    Y = tf.add(tf.matmul(W,X),b)
    
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    
    return result

#1.2 - Computing the sigmoid
def sigmoid(z):
    x = tf.placeholder(tf.float32,name = 'x')
    sigmoid = tf.sigmoid(x,name = 'sigmoid')
    
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict = {x:z})
        
    return result

#1.3 - Computing the Cost
def cost(logits,labels):
    z = tf.placeholder(tf.float32,name = 'z')
    y = tf.placeholder(tf.float32,name = 'y')
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    
    sess = tf.Session()
    cost = sess.run(cost,feed_dict = {z:logits,y:labels})
    sess.close()
    
    return cost

#1.4 - Using One Hot encodings
def one_hot_matrix(labels,C):
    C = tf.constant(value = C,name = 'C')
    one_hot_matrix = tf.one_hot(labels,C,axis = 0)
    
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot

#1.5 - Initialize with zeros and ones
def ones(shape):
    ones = tf.ones(shape)
    
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    
    return ones

###############################################################################
#2 - Building your first neural network in tensorflow

#加载数据集
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

#数据预处理
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


#2.1 - Create placeholders
def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,shape = [n_x,None])
    Y = tf.placeholder(tf.float32,shape = [n_y,None])
    
    return X,Y

#2.2 - Initializing the parameters
def initialize_parameters():
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2,
                  "W3":W3,
                  "b3":b3}
    return parameters

#2.3 - Forward propagation in tensorflow
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3

#2.4 Compute cost
def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))
    
    return cost

#2.5 - Backward propagation & parameter updates

#2.6 - Building the model
def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,
          num_epochs = 1500,minibatch_size = 32,print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    X,Y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost += minibatch_cost/num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f"%(epoch,epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.xlabel('iteration(per ten)')
        plt.ylabel('cost')
        plt.title('Learning_rate='+str(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
        print("Train Accuracy:",accuracy.eval({X:X_train,Y:Y_train}))
        print("Test Accuracy:",accuracy.eval({X:X_test,Y:Y_test}))
        
        return parameters
    
parameters = model(X_train,Y_train,X_test,Y_test)