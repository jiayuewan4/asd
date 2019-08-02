import keras
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential   # Sequential顺序建立
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import RMSprop,Adam
model = Sequential()
## 第一层卷积
model.add(Convolution2D(nb_filter=32,    # 32个filter，即从32个特征提取
                        nb_row=5,        # patch大小
                        nb_col=5, 
                        border_mode='same', 
                        dim_ordering='th',  # theano使用th,TensorFlow使用tf
                        input_shape=(1,28,28,)  # 输入的大小，1表示输入的channel通道，由于是黑白图所以是1,若是rgb是3个通道
                        ))
## 第一层激活层
model.add(Activation('relu'))
## 第一层池化层
model.add(MaxPooling2D(
    pool_size=(2,2),    # 2x2的大小
    strides=(2,2),      # 步长为2，纵向和横向
    border_mode='same'
))
### 第二层卷积层
model.add(Convolution2D(nb_filter=64,      # 不需要指定输入的大小了
                        nb_row=5,
                        nb_col=5, 
                       border_mode='same'
                       ))
### 第二层激活层
model.add(Activation('relu'))
### 第二层池化层
model.add(MaxPooling2D(border_mode='same'))
#### 全连接层
model.add(Flatten())   # 展开
model.add(Dense(output_dim=1024))  # 输出维度为1024
model.add(Activation('relu'))
model.add(Dense(output_dim=10))    # 最终输出为10类
model.add(Activation('softmax'))
adam = Adam()
model.compile(optimizer=adam,     # 使用adam的optimizer
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train)
'''测试集模型'''
loss,accuracy = model.evaluate(X_test,y_test)
print("loss",loss)
print('accuracy',accuracy)
