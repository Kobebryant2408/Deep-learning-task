# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:47:04 2018

@author: kobe24

主要内容:紧接11.1.py的内容,利用keras创建语言模型
"""

#4 - Writing like Shakespeare
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model,load_model,Sequential
from keras.layers import Dense,Activation,Dropout,Input,Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io

print_callback = LambdaCallback(on_epoch_end = on_epoch_end)

model.fit(x,y,batch_size=128,epochs=1,callbacks=[print_callback])

generate_output()