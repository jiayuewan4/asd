__author__ = 'Xiaolin Shen'

from sklearn import svm

import numpy as np

from sklearn import model_selection

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import colors
# 当使用numpy中的loadtxt函数导入该数据集时，假设数据类型dtype为浮点型，但是很明显数据集的第五列的数据类型是字符串并不是浮点型。

# 因此需要额外做一个工作，即通过loadtxt()函数中的converters参数将第五列通过转换函数映射成浮点类型的数据。

# 首先，我们要写出一个转换函数：

# 定义一个函数，将不同类别标签与数字相对应

def iris_type(s):

    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}

    return class_label[s]

 

#（1）使用numpy中的loadtxt读入数据文件

filepath='数据.txt'  # 数据文件路径

data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})

#以上4个参数中分别表示：

#filepath ：文件路径。eg：C:/Dataset/iris.txt。

#dtype=float ：数据类型。eg：float、str等。

#delimiter=',' ：数据以什么分割符号分割。eg：‘，’。

#converters={4:iris_type} ：对某一列数据（第四列）进行某种类型的转换，将数据列与转换函数进行映射的字典。eg：{1:fun}，含义是将第2列对应转换函数进行转换。

#                          converters={4: iris_type}中“4”指的是第5列。

 

print(data)
