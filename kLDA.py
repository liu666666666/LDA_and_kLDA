# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:59:26 2018

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