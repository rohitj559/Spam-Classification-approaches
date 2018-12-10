# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:10:21 2018

@author: Rohit
"""
# =============================================================================
#  Import libraries
# =============================================================================
import os
from bs4 import BeautifulSoup
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =============================================================================
# Process and convert raw email content into text files
# =============================================================================
# Run these only once
def ConvertMailsToText(source, dest):
    sourceContent = os.listdir(source)
    for fname in sourceContent:
        source = os.path.join(source, fname)
        f = open(source, "r")
        html = f.read()
        f.close()
        soup = BeautifulSoup(html, 'html.parser')
        s = soup.get_text()
        s = s.encode('UTF-8')
        if not os.path.exists(dest): # dest path doesnot exist
            os.makedirs(dest)
        f = open(dest + fname + ".txt", "wb")
        f.write(s)
        f.close()

# process training data
sourceTrain = os.getcwd() + "\CSDMC2010_SPAM\TRAINING"
destinationTrain = os.getcwd() + "trainTextMails"
ConvertMailsToText(sourceTrain, destinationTrain)

# process test data
sourceTest = os.getcwd() + "\CSDMC2010_SPAM\TESTING"
destinationTest = os.getcwd() + "testTextMails"
ConvertMailsToText(sourceTest, destinationTest)









# Give your own path for ham and spam mails
ham_path = 'C:\\Users\\Rohith\\Desktop\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\ham'
spam_path = 'C:\\Users\\Rohith\\Desktop\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\spam'  
 
##################################################################################################
# conversion of ham and spam from text to digits  
##################################################################################################  
ham_folder = os.listdir(ham_path)
spam_folder = os.listdir(spam_path)
#remove = dict.fromkeys(map(ord, '\n ' + string.punctuation.replace(' ','')))
punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
digits_dict = {}
digits_list = []
count = 0

# conversion of all ham mails into digits list
ham_list = []
for file in ham_folder:
    source = os.path.join(ham_path, file)
    fin = open(source, "r", encoding="utf-8")
    fin.seek(0,0)
    fin = ''.join(x for x in fin if x not in punctuation)
    fin = fin.lower()
    fin_words = word_tokenize(fin)    
    fin_digits_list = []
    for word in fin_words:
        if word not in digits_dict.keys():
            digits_dict[word] =  count
            count = count + 1
            fin_digits_list.append(digits_dict[word])
        else:
            fin_digits_list.append(digits_dict[word])
# test
# =============================================================================
#     if len(fin_digits_list) < 15:
#         print("Warning! on File: " + str(file));
# =============================================================================
    digits_list.extend(fin_digits_list)
    
# conversion of spam mails into digits dictioanry    
digits_dict2 = {}
digits_list2 = []
count2 = 0

ham_list = []
for file in spam_folder:
    source = os.path.join(spam_path, file)
    fin = open(source, "r", encoding="utf-8")
    fin.seek(0,0)
    fin = ''.join(x for x in fin if x not in punctuation)
    fin = fin.lower()
    fin_words2 = word_tokenize(fin)    
    fin_digits_list2 = []
    for word in fin_words2:
        if word not in digits_dict2.keys():
            digits_dict2[word] =  count2
            count2 = count2 + 1
            fin_digits_list2.append(digits_dict2[word])
        else:
            fin_digits_list2.append(digits_dict2[word])
    if len(fin_digits_list2) < 10:
        print("Warning! on File: " + str(file));
    digits_list2.extend(fin_digits_list2)

# test 
# size of ham numeric list(digits_list) and spam numeric list(digits_list2)
# =============================================================================
# print(len(digits_list));
# print(len(digits_list2));
# =============================================================================

##################################################################################################
# preprocessing to split the digits list obtained above into 32X32X3 chunk matrices
##################################################################################################
unitMatrixSize = 32;
no_of_channels = 3;
# unitSize = (no_of_channels*unitMatrixSize*unitMatrixSize);
unitSize = (unitMatrixSize*unitMatrixSize);
# no of ham 3d matrices, extracting only the integer part
totalHamSamples = int(len(digits_list)/unitSize) 
totalSpamSamples = int(len(digits_list2)/unitSize)

# removing list elements towards the end of the list which cant be converted to matrices
mod = len(digits_list)%unitSize
offset = len(digits_list)-mod
ham_digits_list = digits_list[0:offset]

# generation of ham matrices
newList = []
hamMatrixList = []
for i in range(0, len(ham_digits_list), unitSize):
    newList = ham_digits_list[i:i + unitSize]
    hamMatrix = np.reshape(newList[:] ,(unitMatrixSize, unitMatrixSize))
    hamMatrix3Channeled = np.concatenate([[hamMatrix]] * 3, axis=0)
    hamMatrix3Channeled = np.transpose(hamMatrix3Channeled, (1, 2, 0))
    hamMatrixList.append(hamMatrix3Channeled)
    
# removing list elements towards the end of the list which cant be converted to matrices
mod = len(digits_list2)%unitSize
offset = len(digits_list2)-mod
spam_digits_list = digits_list2[0:offset]

# generation of ham matrices
newList = []
spamMatrixList = []
for i in range(0, len(spam_digits_list), unitSize):
    newList = spam_digits_list[i:i + unitSize]
    spamMatrix = np.reshape(newList[:] ,(unitMatrixSize, unitMatrixSize))
    spamMatrix3Channeled = np.concatenate([[spamMatrix]] * 3, axis=0)
    spamMatrix3Channeled = np.transpose(spamMatrix3Channeled, (1, 2, 0))
    spamMatrixList.append(spamMatrix3Channeled)   

# conversion of lists to arrays
hamArray = np.array(hamMatrixList);
spamArray = np.array(spamMatrixList); 

##################################################################################################
# Extracting features using deep learning models
##################################################################################################
# Training the Resnet model    
from keras.applications.resnet50 import ResNet50
resnet_model = ResNet50(input_shape=(32, 32, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions

# test
# =============================================================================
# print("Total Params:", resnet_model.count_params()) # total no of parameters
# =============================================================================

##################################################################################################
# extract the features for ham and spam seperately
##################################################################################################
ham_FeatureArray = resnet_model.predict(hamArray);
spam_FeatureArray = resnet_model.predict(spamArray);

hamDF = pd.DataFrame(ham_FeatureArray.squeeze())
spamDF = pd.DataFrame(spam_FeatureArray.squeeze())

hamDF["label"] = [0]*len(hamDF)
spamDF["label"] = [1]*len(spamDF)

# test
# =============================================================================
# hamDF["label"].head()
# spamDF["label"].head()
# =============================================================================

frames = [hamDF, spamDF]
df = pd.concat(frames)

df = df.sample(frac=1).reset_index(drop=True)

##################################################################################################
# Splitting the dataset into  Training set and Test set
##################################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'], axis=1), df['label'], test_size=0.2, random_state=0)

# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

##################################################################################################
# Classification models
##################################################################################################
# =============================================================================
# from sklearn.linear_model import LogisticRegression
# #classifier = LogisticRegression(random_state = 0)
# classifier = LogisticRegression(C=166.81005372000593, n_jobs=-1,
#           penalty='l2')
# mod_logistic = classifier.fit(X_train, y_train)
# 
# # Predicting the Validation set results
# y_pred = mod_logistic.predict(X_test)
# 
# print(confusion_matrix(y_test,y_pred))
# print(accuracy_score(y_test,y_pred))
# print(precision_score(y_test, y_pred))
# print(recall_score(y_test,y_pred))
# print(f1_score(y_test,y_pred))
# =============================================================================
##################################################################################################

##################################################################################################
# classification models with grid search and cross validation
##################################################################################################

# Logistic Regression

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

from sklearn import linear_model
# Create logistic regression
logistic = linear_model.LogisticRegression()

# Create grid search using 5-fold cross validation
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, n_jobs=-1)

# Fit grid search
# =============================================================================
# best_model = clf.fit(df.drop(['label'], axis=1), df['label'])
# =============================================================================
best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# prediction on test data using the best model 
y_pred_logistic = best_model.predict(X_test)

print(confusion_matrix(y_test,y_pred_logistic))
print(accuracy_score(y_test,y_pred_logistic))
print(precision_score(y_test, y_pred_logistic))
print(recall_score(y_test,y_pred_logistic))
print(f1_score(y_test,y_pred_logistic))

# =============================================================================
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# =============================================================================

# =============================================================================
# # Calculate accuracy
# accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
# =============================================================================

# =============================================================================
# Best Penalty: l2
# Best C: 166.81005372000593
# =============================================================================

# SVM

from sklearn.svm import LinearSVC

svc = LinearSVC(C=1.0)
model_svm = svc.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)

print(confusion_matrix(y_test,y_pred_svm))
print(accuracy_score(y_test,y_pred_svm))
print(precision_score(y_test, y_pred_svm))
print(recall_score(y_test,y_pred_svm))
print(f1_score(y_test,y_pred_svm))

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on data1's feature and target data
best_model_svmGrid = clf.fit(X_train, y_train) 

# prediction on test data using the best model 
y_pred_svmGrid= best_model_svmGrid.predict(X_test)

print(confusion_matrix(y_test,y_pred_svmGrid))
print(accuracy_score(y_test,y_pred_svmGrid))
print(precision_score(y_test, y_pred_svmGrid))
print(recall_score(y_test,y_pred_svmGrid))
print(f1_score(y_test,y_pred_svmGrid))

# =============================================================================

# Adaboost

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
                         random_state=0)

model = clf.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)

print(confusion_matrix(y_test,y_pred_svm))
print(accuracy_score(y_test,y_pred_svm))
print(precision_score(y_test, y_pred_svm))
print(recall_score(y_test,y_pred_svm))
print(f1_score(y_test,y_pred_svm))
# =============================================================================

# Random Forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1)

model_randomForest = clf.fit(X_train, y_train)

# Make new observation
observation = [[ 5,  4,  3,  2]]
              
# Predict observation's class    
y_pred_randomForest = model_randomForest.predict(X_test)

print(confusion_matrix(y_test,y_pred_randomForest))
print(accuracy_score(y_test,y_pred_randomForest))
print(precision_score(y_test, y_pred_randomForest))
print(recall_score(y_test,y_pred_randomForest))
print(f1_score(y_test,y_pred_randomForest))
# =============================================================================

# Neural Network:
from keras.models import Sequential
from keras.layers import Dense

classifierAN = Sequential()
classifierAN.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 512))
classifierAN.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifierAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifierAN.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_predAN = classifierAN.predict(X_test)

print(confusion_matrix(y_test,y_predAN))
print(accuracy_score(y_test,y_predAN))
print(precision_score(y_test, y_predAN))
print(recall_score(y_test,y_predAN))
print(f1_score(y_test,y_predAN))


classifierAN = Sequential()
classifierAN.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu', input_dim = 2048))
classifierAN.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
classifierAN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifierAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifierAN.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_predAN = classifierAN.predict(X_test)

print(confusion_matrix(y_test,y_predAN))
print(accuracy_score(y_test,y_predAN))
print(precision_score(y_test, y_predAN))
print(recall_score(y_test,y_predAN))
print(f1_score(y_test,y_predAN))




