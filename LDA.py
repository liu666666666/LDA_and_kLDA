# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:45:52 2018

@author: 11854
"""
import numpy as np
import matplotlib.pyplot as plt
#读取数据
filename='data_LDA.txt'
train_data = np.loadtxt(filename, delimiter=',', dtype=np.str)
train_data = np.float64(train_data)

data_num=len(train_data)
class_num=2
each_class_num=data_num/class_num
each_class_num=int(each_class_num)
m=np.zeros([2,2])
m0=np.mean(train_data,axis=0)

S_LDA_w=np.zeros([2,2])
#求各个类的均值,求类内方差矩阵S_LDA_w
for i in range(class_num):
    temp_data=train_data[i*each_class_num:(i+1)*each_class_num,:]
    m[i]=np.mean(temp_data,axis=0)
    S_LDA_w=S_LDA_w+np.dot((temp_data-m[i]).T,(temp_data-m[i]))/data_num
    
    #m的每一行是一个类的均值


#求类间的方差S_LDA_b,由于每类样本数一样，所以这里可以简化计算
S_LDA_b=np.zeros([2,2])
S_LDA_b=S_LDA_b+np.dot((m-m0).T,(m-m0))/2

#求特征值，特征向量
eigenvalue,featurevector=np.linalg.eig(np.linalg.inv(S_LDA_w).dot(S_LDA_b))

#求最大特征值对应的特征向量
newspace=featurevector[:,np.argmax(eigenvalue)]

##new_data_1=(train_data[0:201,:]).dot(newspace)
#new_data_2=(train_data[201:402,:]).dot(newspace)

k=newspace[1]/newspace[0]
new_data=train_data.dot(newspace)
new_data_map=np.zeros_like(train_data)
for i in range(data_num):
    new_data_map[i][0]=train_data[i].dot([[1],[k]])/(k*k+1)
    new_data_map[i][1]=new_data_map[i][0]*k

#画图
plt.scatter(train_data[0:201,0], train_data[0:201,1], s=5,c='g')
plt.scatter(train_data[201:402,0], train_data[201:402,1], s=5,c='b')   
    
plt.scatter(new_data_map[0:201,0], new_data_map[0:201,1], s=5,c='g')
plt.scatter(new_data_map[201:402,0], new_data_map[201:402,1], s=5,c='b')
plt.show()     



