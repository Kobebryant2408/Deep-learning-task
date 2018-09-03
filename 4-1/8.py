# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 19:13:03 2018

@author: kobe24

特别注意:熟悉conv层与pool层的前向传播的推导,了解其后向传播的推导
"""

###############################################################################
#Part 1：Convolutional Neural Networks: Step by Step
#1 - Packages
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#2 - Outline of the Assignment

#3 - Convolutional Neural Networks(conv layer)
#3.1 - Zero-Padding
def zero_pad(X,pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    
    return X_pad

#3.2 - Single step of convolution
def conv_single_step(a_slice_prev,W,b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
    
    return Z

#3.3 - Convolutional Neural Networks - Forward pass
def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    
    Z = np.zeros((m,n_H,n_W,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride*h
                    vert_end = vert_start + f
                    horiz_start = stride*w
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
                    
    assert(Z.shape == (m,n_H,n_W,n_C))
    cache = (A_prev,W,b,hparameters)
    
    return Z,cache

#4 - Pooling layer
#4.1 - Forward Pooling
def pool_forward(A_prev,hparameters,mode="max"):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int((n_H_prev - f)/stride + 1)
    n_W = int((n_W_prev - f)/stride + 1)
    n_C = n_C_prev
    
    A = np.zeros((m,n_H,n_W,n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice_prev)
    
    cache = (A_prev,hparameters)
    return A,cache

#5 - Backpropagation in convolutional neural networks (OPTIONAL / UNGRADED)
#5.1 - Convolutional layer backward pass
#5.1.1 - Computing dA:da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
#5.1.2 - Computing dW:dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
#5.1.3 - Computing db:db[:,:,:,c] += dZ[i, h, w, c]
def conv_backward(dZ,cache):
    (A_prev,W,b,hparameters) = cache
    
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m,n_H,n_W,n_C) = dZ.shape
    
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
        
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]
        
    assert(dA_prev.shape == (m,n_H_prev,n_W_prev,n_C_prev))
    
    return dA_prev,dW,db

#5.2 Pooling layer - backward pass
#5.2.1 Max pooling - backward pass
def create_mask_from_window(x):
    mask = (x == np.max(x))
    
    return mask

#5.2.2 - Average pooling - backward pass
def distribute_value(dz,shape):
    (n_H,n_W) = shape
    average = dz / (n_H * n_W)
    a = average * np.ones(shape)
    
    return a

#5.2.3 Putting it together: Pooling backward
def pool_backward(dA, cache, mode = "max"):

    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(np.shape(A_prev))

    for i in range(m):                       

        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])


                    elif mode == "average":

                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev
