# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:57:07 2018

@author: Rohit
"""
import pandas as pd
import numpy as np
import os
from nltk import word_tokenize
from numpy import array
import math

# =============================================================================
# data = pd.read_csv('spam.csv', encoding='latin1')
# 
# #Drop column and name change
# data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# data = data.rename(columns={"v1":"label", "v2":"text"})
# =============================================================================


srcDirectory = 'C:/Users/Rohith/Desktop/Fall_2018/Research/Exercise8_Spam_Play_with_text/Processed_mails/TRAINING'
files = os.listdir(srcDirectory)

punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
digits_dict = {}
digits_list = []
count = 0

# pre-processing
for file in files:
    srcpath = os.path.join(srcDirectory, file)
    row = open(srcpath, encoding="utf8")
    row.seek(0,0)
    #row = row.read()
    row = ''.join(x for x in row if x not in punctuation) # exclude punctuations
    row = row.lower()
    row_words = word_tokenize(row)  
    row_digits_list = []
    for word in row_words:
        if word not in digits_dict.keys():
            digits_dict[word] =  count
            count = count + 1
            row_digits_list.append(digits_dict[word])
        else:
            row_digits_list.append(digits_dict[word])
    digits_list.append(row_digits_list)
    
# =============================================================================
# count = 0
# for row in digits_list:
#     for digit in row:
#         count = count + 1
# =============================================================================

max_length = max(len(l) for l in digits_list)
max_length_root = int(math.sqrt(max_length))
max_length_perfect_sq_root = int(max_length_root + 1) 

# max_length = 18576 = 129*144
base = np.zeros(max_length_perfect_sq_root**2)


# =============================================================================
# if(max_length % 2 == 0):
#     l = int(max_length/2)
#     base = np.zeros(max_length)
# else:
#     base = np.zeros(max_length+1)
#     l = int((max_length+1)/2)
# =============================================================================
 
temp_array = (base.reshape(max_length_perfect_sq_root, max_length_perfect_sq_root))
  
doclist = []
def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l
    
for n, i in enumerate(digits_list):
    if(len(i) % 2 != 0):
        i.append(0)
        
    #l = int(len(i))
    l_root = int(math.sqrt(len(i)))
    #difference = abs(l_root**2 - len(i))
    perfect_sq_root = int(l_root + 1)    

    new_list = pad(i, 0, (perfect_sq_root**2))
    #new_list_sqrt = int(math.sqrt(len(new_list)))    
    
    #try:
    temp_line = array(new_list).reshape((perfect_sq_root, perfect_sq_root))
# =============================================================================
#     except Exception as e:
#         print("index = {0}", n)
#         template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#         message = template.format(type(e).__name__, e.args)
#         print (message)
# =============================================================================
        
    temp_array[:temp_line.shape[0],:temp_line.shape[1]] = temp_line
    dataframe = pd.DataFrame.from_records(temp_array)
    doclist.append(dataframe)   

#from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import vgg16
def extract_resnet(X):  
    # X : images numpy array
    #resnet_model = ResNet50(input_shape=(32, 32, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions
    vgg_model = vgg16(input_shape=(10, 10, 3), weights='imagenet', include_top=False)
    features_array = resnet_model.predict(X)
    return features_array
     
