# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:57:25 2018

@author: kobe24

遇到的问题:(3.1)第三部分利用链式法则推导反向传播时,计算公式到底是点乘(.)还是对应元素相乘(*)搞不清楚,
          比如,dtanh计算为*乘,dWax,dWaa,...的计算为.乘;(尤其是dtanh的计算)
          (3.1)rnn_backward函数中,调用rnn_cell_backward函数时,第一个参数为什么要+da_prevt;
          (3.2,3.3没有做)
主要内容:1和3.1分别为标准RNN的前后向传播(基于RNN unit);
        2和3.2分别为基于LSTM unit的RNN的前后向传播.
        
"""

#Building your Recurrent Neural Network - Step by Step
import numpy as np
from rnn_utils import *

#1 - Forward propagation for the basic Recurrent Neural Network
#1.1 - RNN cell(单个时间步t)
def rnn_cell_forward(xt, a_prev, parameters): #a_prev = a <t-1>
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    a_next = np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba) #a_next = a <t>
    yt_pred = softmax(np.dot(Wya,a_next)+by)               #yt_pred = y hat <t>
    
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

#1.2 - RNN forward pass(T_x个时间步)
def rnn_forward(x, a0, parameters):
    caches = []
    
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    a_next = a0
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches

#2 - Long Short-Term Memory (LSTM) network
#2.1 - LSTM cell
def lstm_cell_forward(xt, a_prev, c_prev, parameters): #a_prev = a <t-1>
    Wf =parameters["Wf"]                               #c_prev = c <t-1>
    bf =parameters["bf"]
    Wi =parameters["Wi"]
    bi =parameters["bi"]
    Wo =parameters["Wo"]
    bo =parameters["bo"]
    Wc =parameters["Wc"]
    bc =parameters["bc"]
    Wy =parameters["Wy"]
    by =parameters["by"]
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    
    concat = np.zeros((n_a+n_x,m))
    concat[:n_a,:] = a_prev
    concat[n_a:,:] = xt
    
    ft = sigmoid(np.dot(Wf,concat)+bf) #forget gate
    it = sigmoid(np.dot(Wi,concat)+bi) #update gate
    cct = np.tanh(np.dot(Wc,concat)+bc) #c tilde <t>
    c_next = ft*c_prev + it*cct         #c <t>
    ot = sigmoid(np.dot(Wo,concat)+bo) #output gate
    a_next = ot*np.tanh(c_next)         #a <t>
    
    yt_pred = softmax(np.dot(Wy,a_next)+by) #y hat <t>
    
    cache = (a_next,c_next,a_prev,c_prev,ft,it,cct,ot,xt,parameters)
    
    return a_next,c_next,yt_pred,cache

#2.2 - Forward pass for LSTM
def lstm_forward(x,a0,parameters):
    caches = []
    
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        a[:,:,t] = a_next
        c[:,:,t] = c_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    
    caches = (caches, x)
    
    return a, y_pred, c, caches

#3 - Backpropagation in recurrent neural networks (OPTIONAL / UNGRADED)
#3.1 - Basic RNN backward pass
#Deriving the one step backward functions:(一个时间步)
def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    dtanh = (1 - a_next**2) * da_next  #dtanh到底是不是代表dJ/du,其中a<t>=tanh(u) ??????????
    
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    
    dba = np.sum(dtanh, keepdims=True, axis=-1)
    
    gradients = {"dxt":dxt, "da_prev":da_prev, "dWax":dWax, "dWaa":dWaa, "dba":dba}
    
    return gradients

#Backward pass through the RNN:(T_x个时间步)
def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    dx = np.zeros((n_x,m,T_x))
    dWax = np.zeros((n_a,n_x))
    dWaa = np.zeros((n_a,n_a))
    dba = np.zeros((n_a,1))
    da0 = np.zeros((n_a,m))
    da_prevt = np.zeros((n_a,m))
    
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt,caches[t]) #为什么要加da_prevt??????????
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"],gradients["da_prev"],gradients["dWax"],gradients["dWaa"],gradients["dba"]
        dx[:,:,t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    da0 = da_prevt
    
    gradients = {"dx":dx,"da0":da0,"dWax":dWax,"dWaa":dWaa,"dba":dba}
    
    return gradients

#3.2 - LSTM backward pass
#3.2.1 One Step backward
#3.2.2 gate derivatives
#3.2.3 parameter derivatives

#3.3 Backward pass through the LSTM RNN