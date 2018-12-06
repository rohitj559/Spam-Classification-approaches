# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:15:40 2018

@author: Rohit
"""

import numpy as np
#rand = np.random.RandomState(25)

x = list(range(144))
print(x)
count = 0
list_of_arrays = []
for i in range(3):
    list_size = count+48;
    temp_array =  np.reshape(x[count:list_size],(3, 4, 4));
    list_of_arrays.append(temp_array);
    count = list_size;
    
A = np.arange(27)
B = np.reshape(A,(3,3,3))
print(B[:,:,:])

data = [[[1, 2, 3, 4],
		 [5, 6, 7, 8],
		 [5, 6, 7, 8],
         [9, 10, 11, 12]],
    
        [[13, 14, 15, 16],
		 [17, 18, 19, 20],
		 [21, 22, 23, 24],
         [25, 26, 27, 28]],
         
        [[29, 30, 31, 32],
		 [33, 34, 35, 36],
		 [37, 38, 39, 40],
         [41, 42, 43, 44]]]

data = np.array(data)
print(data.shape)

import numpy as np
arrays = [np.random.randn(3, 3) for _ in range(3)]

arrayStacked = np.stack(arrays, axis=-1)

array = 

nchannels = 3;
height = 3;
width = 3;

array = np.reshape(list(range(1, 10)), (height, width))

X = np.concatenate([[array]] * 3, axis=0)

arrayZeros = np.zeros((height,width,nchannels))
for ch in range(nchannels):
    for h in range(height):
        for w in range(width):
            arrayZeros[h,w,ch] = array[height, width]
stackedArray = arrayZeros
print(stackedArray)
            


# =============================================================================
# # if we need 32X32X3 matrices of ham and spam lists:
# unitMatrixSize = 32;
# no_of_channels = 3;
# unitSize = (unitMatrixSize*unitMatrixSize*no_of_channels);
# # no of ham 3d matrices, extracting only the integer part
# ham_samples = 326
# spam_samples = int(len(digits_list2)/unitSize)
# 
# # generation of ham matrices
# 
# count = 0
# list_of_ndarrays = []
# for i in range(ham_samples):
#     list_size = count + unitSize;
#     temp_array =  np.reshape(digits_list[count:list_size],(unitMatrixSize,unitMatrixSize,no_of_channels));
#     list_of_ndarrays.append(temp_array);
#     count = list_size;
# 
#     
# =============================================================================
