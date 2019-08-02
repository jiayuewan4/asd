#!/usr/bin/python
# -*- coding:utf8 -*-
'''
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# pretreatment 预处理
# input image dimensions
img_rows, img_cols = 28, 28

import numpy as np
y_train = np.loadtxt(open("train.csv",'rb'),delimiter=',',skiprows=1,usecols=0,max_rows=40000)
x_train = np.loadtxt(open('train.csv','rb'),delimiter=',',skiprows=1,usecols=range(784),max_rows=40000)
y_test = np.loadtxt(open('train.csv','rb'),delimiter=',',skiprows=40001,usecols=0)
x_test = np.loadtxt(open('train.csv','rb'),delimiter=',',skiprows=40001,usecols=range(784))




if K.image_data_format() == 'channels_first': #判断图片格式是 channel在前还是在后（channel：黑白为1,彩色为3）
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) #shape[0]指例子的个数
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# build the neural net 建模型(卷积—relu-卷积-relu-池化-relu-卷积-relu-池化-全连接)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) # 32个过滤器，过滤器大小是3×3，32×26×26
model.add(Conv2D(64, (3, 3), activation='relu')) #64×24×24
model.add(MaxPooling2D(pool_size=(2, 2)))# 向下取样
model.add(Dropout(0.25))
model.add(Flatten()) #降维：将64×12×12降为1维（即把他们相乘起来）
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) #全连接2层


# compile the model 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train the model 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

# test the model 测试模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

