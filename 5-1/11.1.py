# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:00:22 2018

@author: kobe24

主要内容:利用numpy构建基于字符的语言模型,
        利用恐龙名字数据集进行训练,然后对训练完的模型进行采样

特别注意:训练样本的构造(113-115),x<t+1>=y<t>,n_x=n_y=vocab_size;
        由于一个训练样本对应一个名字序列,优化算法采用随机梯度下降(SGD);

"""

#Character level language model - Dinosaurus land
import numpy as np
from utils import *
import random

#1 - Problem Statement
#1.1 - Dataset and Preprocessing
data = open('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（5-1）Part 2 - Character level language model - Dinosaurus land/dataset/dinos.txt','r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print('There are %d total characters and %d unique characters in your data.'% (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars))}
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars))}
#print(ix_to_char)

#1.2 - Overview of the model

#2 - Building blocks of the model
#2.1 - Clipping the gradients in the optimization loop
def clip(gradients, maxValue):
    dWaa,dWax,dWya,db,dby = gradients['dWaa'],gradients['dWax'],gradients['dWya'],gradients['db'],gradients['dby']
    
    for gradient in [dWaa,dWax,dWya,db,dby]:
        np.clip(gradient,-maxValue,maxValue,out=gradient)
        
    gradients = {"dWaa":dWaa,"dWax":dWax,"dWya":dWya,"db":db,"dby":dby}
    
    return gradients

#2.2 - Sampling
def sample(parameters, char_to_ix, seed):
    Waa,Wax,Wya,by,b = parameters["Waa"],parameters["Wax"],parameters["Wya"],parameters["by"],parameters["b"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    
    indices = []
    idx = -1
    
    counter = 0
    newline_character = char_to_ix['\n']
    while(idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        
        np.random.seed(counter+seed)
        
        idx = np.random.choice(range(len(y)),p=y.ravel())
        
        indices.append(idx)
        
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        a_prev = a
        
        seed += 1
        counter += 1
        
    if (counter == 50):
        indices.append(char_to_ix['\n'])
        
    return indices

#3 - Building the language model
#3.1 - Gradient descent
def optimize(X,Y,a_prev,parameters,learning_rate=0.01):
    loss,cache = rnn_forward(X,Y,a_prev,parameters)
    
    gradients,a = rnn_backward(X,Y,parameters,cache)
    
    gradients = clip(gradients,5)
    
    parameters = update_parameters(parameters,gradients,learning_rate)
    
    return loss,gradients,a[len(X)-1]

#3.2 - Training the model
def model(data,ix_to_char,char_to_ix,num_iterations=35000,n_a=50,dino_names=7,vocab_size=27):
    n_x, n_y = vocab_size, vocab_size
    
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    loss = get_initial_loss(vocab_size, dino_names)
    
    with open("E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（5-1）Part 2 - Character level language model - Dinosaurus land/dataset/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    np.random.seed(0)
    np.random.shuffle(examples)
    
    a_prev = np.zeros((n_a,1))
    
    for j in range(num_iterations):
        #use the hint above to define one training example (X,Y) 
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        
        #前向传播，代价计算,后向传播,梯度修剪以及参数更新(通过optimize函数实现)
        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,learning_rate=0.01)
        
        loss = smooth(loss,curr_loss)
        
        if j % 2000 == 0:
            print("Iteration: %d, Loss: %f" % (j, loss) + '\n')
            
            seed = 0
            #采样得到新的序列数(dino_names)
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
            print('\n')
            
    return parameters
parameters = model(data,ix_to_char,char_to_ix)