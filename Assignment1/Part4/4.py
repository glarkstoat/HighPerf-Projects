# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:52:36 2020
@author: Christian Wiskott
"""
import numpy as np
import numpy.random.uniform as uni
# 14.1
arr1 = np.random.uniform(-1,1,3)
rand1 = arr1 / np.linalg.norm(arr1)

#14.2
def rand2(arr):
    if np.linalg.norm(arr) > 1:
        return rand2(arr)
    else:
        return arr / np.linalg.norm(arr)
    
arr2 = rand2(arr1)

#14.3 

phi = np.random.uniform()