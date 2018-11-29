# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:42:02 2018

@author: Rohit
"""

import pandas as pd
import numpy as np
from nltk import word_tokenize
from numpy import array

data = pd.read_csv('spam.csv', encoding='latin1')

#Drop column and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
digits_dict = {}
digits_list = []
count = 0

# pre-processing
for row in data['text']:
    #row = ''.join(x for x in row if x not in punctuation) # exclude punctuations
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
    
count = 0
for row in digits_list:
    for digit in row:
        count = count + 1

max_length = max(len(l) for l in digits_list)

if(max_length % 2 == 0):
    l = int(max_length/2)
    base = np.zeros(max_length)
else:
    base = np.zeros(max_length+1)
    l = int((max_length+1)/2)
 
temp_array = (base.reshape(l, -1))
  
doclist = []    
for n, i in enumerate(digits_list):
    if(len(i) % 2 == 0):
        l = int(len(i)/2)
    else:
        i.append(0)
    l = int((len(i)+1)/2)
    
    try:
        temp_line = (array(i)).reshape(l, -1)
    except Exception as e:
        print("index = {0}", n)
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        
    temp_array[:temp_line.shape[0],:temp_line.shape[1]] = temp_line
    dataframe = pd.DataFrame.from_records(temp_array)
    doclist.append(dataframe)   

from keras.applications.resnet50 import ResNet50
def extract_resnet(X):  
    # X : images numpy array
    resnet_model = ResNet50(input_shape=(108, 2, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(X)
    return features_array
     
