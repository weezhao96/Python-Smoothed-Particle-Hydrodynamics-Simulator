# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:43:03 2022

@author: Zhao
"""
import numpy as np

x = np.array([0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5,
              0.5, 0.5, 0.6, 0.5, 0.7, 0.5, 0.8, 0.5, 0.9, 0.5])


print(x.shape)

y = np.ndarray(9, dtype = np.float64,
               buffer=x, offset=8,
               strides=16)

print(y)




def func(x):
    x = 0.0

z = 1
func(z)
print(z)
