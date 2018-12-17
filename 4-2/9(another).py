# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:01:42 2018

@author: kobe24

主要内容:是一个有监督的二类分类问题.
        相较于9.py利用keras构建的CNN模型,9(another).py构建的模型深度更深,测试正确率更高.
        其实就是将(Conv-BN-activation-MaxPool)这四层结构重复了三次,只是其中的超参数选取不同.
        
特别注意:9(another).py与9.1(another).py主要是为了熟悉keras搭建函数式模型(Model)的步骤;
        以及掌握训练过程与测试过程的可视化方法(其中9(another).py直接利用fit函数返回的History对象;
        9.1(another).py则定义了继承了keras.callbacks.Callback类的新的回调函数);
        以及了解h5py文件存放数据集的格式.
"""

#1.加载数据
import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
 
    test_dataset = h5py.File('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-2）- Keras tutorial - the Happy House & Residual Networks/dataset/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
     
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes = load_dataset()

X_train = train_set_x_orig / 255
X_test = test_set_x_orig / 255

Y_train = train_set_y_orig.T
Y_test = test_set_y_orig.T

m_train = X_train.shape[0]
num_px = X_train.shape[1]
num_py = X_train.shape[2]
num_pz = X_train.shape[3]
m_test = X_test.shape[0]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Each input is of size: (" + str(num_px) + ", " + str(num_py) + ", " + str(num_pz) + ")")
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#2.利用keras构建模型
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#2.1 搭建模型
def happymodel(input_shape):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3,3))(x_input)
    
    x = Conv2D(32,(3,3),strides=(1,1),name='conv0')(x)
    x = BatchNormalization(axis=3,name='bn0')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='max_pool0')(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),name='conv1')(x)
    x = BatchNormalization(axis=3,name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='max_pool1')(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),name='conv2')(x)
    x = BatchNormalization(axis=3,name='bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2),name='max_pool2')(x)
    
    x = Flatten()(x)
    x = Dense(1,activation='sigmoid',name='fc0')(x)
    
    model = Model(inputs=x_input,outputs=x,name='happymodel')
    return model

model = happymodel(X_train.shape[1:])
#2.2 编译模型
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,loss='binary_crossentropy',
              metrics=['accuracy'])
#2.3 训练模型
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_lr=0.00001)
History = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=20,
                    validation_split=0.2,callbacks=[reduce_lr])
#2.4 评估模型
fig = plt.figure()
plt.plot(History.history['acc'],'r',label='train acc')
plt.plot(History.history['loss'],'g',label='train loss')
plt.plot(History.history['val_acc'],'b',label='val acc')
plt.plot(History.history['val_loss'],'k',label='val loss')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('acc-loss')
plt.legend(loc='upper right')
plt.show()
#2.5 测试模型
preds = model.evaluate(x=X_test,y=Y_test)
print("test loss = "+str(preds[0]))
print("test accuracy = "+str(preds[1]))
model.summary()