# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:57:40 2018

@author: 11854
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:59:26 2018

@author: 11854
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
dim=2

#读取数据
filename='data_LDA.txt'
train_data = np.loadtxt(filename, delimiter=',', dtype=np.str)
train_data = np.float64(train_data)


data_mean=train_data

nun_of_data=data_mean.shape[0]
distance_maxtrix=np.zeros([nun_of_data,nun_of_data])
K=np.zeros([nun_of_data,nun_of_data])
K0=np.zeros([nun_of_data,nun_of_data])
#k是kernel矩阵
kernel_sigma=7







for i in range(nun_of_data):
    for j in range(nun_of_data):
        distance_maxtrix[i][j]=np.linalg.norm(data_mean[i]-data_mean[j])
        K0[i][j]=math.exp(-distance_maxtrix[i][j]**2/2/kernel_sigma**2)


#K0=np.exp(-distance_maxtrix*distance_maxtrix/2/kernel_sigma**2)





#算距离矩阵
#K0=np.exp(-distance_maxtrix*sigma_kernel)
oneN=np.ones_like(K0)/nun_of_data
#oneN=np.ones_like(K0)
K=K0-oneN.dot(K0)-K0.dot(oneN)+(oneN.dot(K0)).dot(oneN)
B=np.zeros_like(K)
B[0:201,0:201]=np.ones((201,201))/201
B[201:402,201:402]=np.ones((201,201))/201

#提取特征值，特征向量
ddd=np.linalg.inv(K0.dot(K0))
ccc=(K0.dot(B)).dot(K0)
#ccc=((np.linalg.inv(K0)).dot(B)).dot(K0)
eigenvalue,featurevector=np.linalg.eig(ddd.dot(ccc))
featurevector=featurevector.real
eigenvalue=eigenvalue.real


eigenvalue_index=np.argsort(-eigenvalue)
eigenvalue_index=eigenvalue_index[0:dim]
#按照降序排列，返回索引
#eigenvalue_sort1=eigenvalue[np.argsort(-eigenvalue)]
eigenvalue_sort=eigenvalue[eigenvalue_index[0:dim]]
featurevector_sort=featurevector[:,eigenvalue_index[0:dim]]

data_mapped=(featurevector_sort.T).dot(K0)
data_mapped=data_mapped.T
plt.scatter(data_mapped[0:201,0], data_mapped[0:201,1], s=5,c='g')
plt.scatter(data_mapped[201:402,0], data_mapped[201:402,1], s=5,c='r')
#plt.scatter(data_mapped[0:201,0], data_mapped[0:201,0], s=5,c='g')
#plt.scatter(data_mapped[201:402,0], data_mapped[201:402,0], s=5,c='r')
plt.show() 



