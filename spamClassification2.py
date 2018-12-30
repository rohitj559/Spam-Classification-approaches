# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 07:29:37 2018

@author: Rohit

Steps involved:
    1. convert all mails into digits 
    2. digit representation of each mail needs to be converted as 32X32 representation
    3. segregate into ham, spam amd test mails
    4. convert all the digits into 32X32 matrices
    5. exclude the left overs
    6. build machine learing models
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
from sklearn.model_selection import GridSearchCV
from itertools import cycle, islice

# Importing classification model librares
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense

# map the inputs to the function blocks
options = {0 : "Logistic_Regression",
           1 : "SVM",
           2 : "K_Nearest_Neighbors",
           3 : "Naive_Bayes",
           4 : "Decision_Tree",
           5 : "Random_Forest"
          }

# =============================================================================
# Process and convert raw email content into text files
# =============================================================================
# Run these only once. The below code needs more edits
# =============================================================================
# def ConvertMailsToText(source, dest):
#     sourceContent = os.listdir(source)
#     for fname in sourceContent:
#         source = os.path.join(source, fname)
#         f = open(source, "r")
#         html = f.read()
#         f.close()
#         soup = BeautifulSoup(html, 'html.parser')
#         s = soup.get_text()
#         s = s.encode('UTF-8')
#         if not os.path.exists(dest): # dest path doesnot exist
#             os.makedirs(dest)
#         f = open(dest + fname + ".txt", "wb")
#         f.write(s)
#         f.close()
# 
# # process training data
# sourceTrain = os.getcwd() + "\CSDMC2010_SPAM\TRAINING"
# destinationTrain = os.getcwd() + "\trainTextMails"
# ConvertMailsToText(sourceTrain, destinationTrain)
# 
# # process test data
# sourceTest = os.getcwd() + "\CSDMC2010_SPAM\TESTING"
# destinationTest = os.getcwd() + "\testTextMails"
# ConvertMailsToText(sourceTest, destinationTest)
# =============================================================================

# =============================================================================
# conversion of ham and spam from text to digits  
# =============================================================================
def MapWordsToDigits(source, path):    
    punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
    digits_dict = {}
    digits_list = []
    count = 0
    
    for file in source:
        sourceFile = os.path.join(path, file)
        fin = open(sourceFile, "r", encoding="utf-8")
        fin.seek(0,0)
        fin = ''.join(x for x in fin if x not in punctuation)
        fin = fin.lower()
        #fin_words1 = word_tokenize(fin)   
        fin_words = fin.split()
        #fin_length = len(fin_words)
        fin_words = list(islice(cycle(fin_words), 1024)) # setting the length of words to 32X32=1024 by re-parsing a given words collection from beginning
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
        digits_list.append(fin_digits_list) # using append method since we need to output digit representation for mails seperately
    return digits_list

def EliminateExtraDigits(lst):
    # removing list elements towards the end of the list which cant be converted to matrices
    mod = len(lst)%(unitMatrixSize*unitMatrixSize)
    offset = len(lst)-mod
    return offset

def GenerateMatrix(dig_list, unitMatrixSize, channels):
    #newList = []
    matrixList = []
    for i in range(0, len(dig_list)):
        #newList = dig_list[i:i + (unitMatrixSize*unitMatrixSize)]
        matrix = np.reshape(dig_list[i] ,(unitMatrixSize, unitMatrixSize))
        matrix3Channeled = np.concatenate([[matrix]] * channels, axis=0)
        matrix3Channeled = np.transpose(matrix3Channeled, (1, 2, 0))
        matrixList.append(matrix3Channeled)
    return matrixList

# function blocks for training and predicting the scores
def Logistic_Regression(X_train, X_test, y_train, y_test):
    classifierL = LogisticRegression(random_state = 0)
    classifierL.fit(X_train, y_train)
    y_predL = classifierL.predict(X_test)
    cmL = confusion_matrix(y_test, y_predL)
    accuracyL = (cmL[0][0] + cmL[1][1]) / (cmL[0][0] + cmL[0][1] + cmL[1][0] + cmL[1][1])
    return accuracyL

def SVM(X_train, X_test, y_train, y_test):
    classifierSVM = LinearSVC(random_state = 0)
    classifierSVM.fit(X_train, y_train)
    y_predL = classifierSVM.predict(X_test)
    cmSVM = confusion_matrix(y_test, y_predL)
    accuracySVM = (cmSVM[0][0] + cmSVM[1][1]) / (cmSVM[0][0] + cmSVM[0][1] + cmSVM[1][0] + cmSVM[1][1])
    return accuracySVM

def K_Nearest_Neighbors(X_train, X_test, y_train, y_test):
    classifierK = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)    
    classifierK.fit(X_train, y_train)
    y_predK = classifierK.predict(X_test)
    cmK = confusion_matrix(y_test, y_predK)
    accuracyK = (cmK[0][0] + cmK[1][1]) / (cmK[0][0] + cmK[0][1] + cmK[1][0] + cmK[1][1])
    return accuracyK
    
def Naive_Bayes(X_train, X_test, y_train, y_test):
    classifierN = GaussianNB()
    classifierN.fit(X_train, y_train)
    y_predN = classifierN.predict(X_test)
    cmN = confusion_matrix(y_test, y_predN)
    accuracyN = (cmN[0][0] + cmN[1][1]) / (cmN[0][0] + cmN[0][1] + cmN[1][0] + cmN[1][1])
    return accuracyN
    
def Decision_Tree(X_train, X_test, y_train, y_test):
    classifierD = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifierD.fit(X_train, y_train)
    y_predD = classifierD.predict(X_test)
    cmD = confusion_matrix(y_test, y_predD)
    accuracyD = (cmD[0][0] + cmD[1][1]) / (cmD[0][0] + cmD[0][1] + cmD[1][0] + cmD[1][1])
    return accuracyD

def Random_Forest(X_train, X_test, y_train, y_test):
    classifierR = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifierR.fit(X_train, y_train)
    y_predR = classifierR.predict(X_test)
    cmR = confusion_matrix(y_test, y_predR)
    accuracyR = (cmR[0][0] + cmR[1][1]) / (cmR[0][0] + cmR[0][1] + cmR[1][0] + cmR[1][1])
    return accuracyR

# For Training
ham_path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\ham'
spam_path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\spam'
ham_folder = os.listdir(ham_path)
spam_folder = os.listdir(spam_path)

# For Testing  
test_Path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\test' 
test_Folder = os.listdir(test_Path)

hamList = MapWordsToDigits(ham_folder, ham_path)
spamList = MapWordsToDigits(spam_folder, spam_path)
testList = MapWordsToDigits(test_Folder, test_Path)
    
# sanity test 
# size of ham numeric list(digits_list) and spam numeric list(digits_list2)
# =============================================================================
# print(len(hamList));
# print(len(spamList));
# =============================================================================

# =============================================================================
# preprocessing to split the digits list obtained above into 32X32X3 chunk matrices
# =============================================================================

# setting dimentions on input for training
unitMatrixSize = 32; # used to generate 32X32 matrices
no_of_channels = 3; # Three channel augmentation

# sanity test
# no of ham 3d matrices, extracting only the integer part
# =============================================================================
# totalHamSamples = int(len(hamList)/unitSize) 
# totalSpamSamples = int(len(spamList)/unitSize)
# =============================================================================

# =============================================================================
# ham_digits_list = hamList[0:EliminateExtraDigits(hamList)]
# spam_digits_list = spamList[0:EliminateExtraDigits(spamList)]
# test_digits_list = testList[0:EliminateExtraDigits(testList)]
# =============================================================================

# generation of matrices
hamArray = np.array(GenerateMatrix(hamList, unitMatrixSize, no_of_channels));
spamArray = np.array(GenerateMatrix(spamList, unitMatrixSize, no_of_channels)); 
testArray = np.array(GenerateMatrix(testList, unitMatrixSize, no_of_channels)); 

# =============================================================================
# Extracting features using deep learning models
# =============================================================================

# Training the Resnet model    
from keras.applications.resnet50 import ResNet50
resnet_model = ResNet50(input_shape=(32, 32, 3), weights='imagenet', include_top=False)  # Since top layer is the fc layer used for predictions

# sanity test
# =============================================================================
# print("Total Params:", resnet_model.count_params()) # total no of parameters
# =============================================================================

# =============================================================================
# extraction the features 
# =============================================================================

ham_FeatureArray = resnet_model.predict(hamArray);
spam_FeatureArray = resnet_model.predict(spamArray);
test_FeatureArray = resnet_model.predict(testArray);

hamDF = pd.DataFrame(ham_FeatureArray.squeeze())
spamDF = pd.DataFrame(spam_FeatureArray.squeeze())
testDF = pd.DataFrame(test_FeatureArray.squeeze())

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

# =============================================================================
# Splitting the dataset into  Training set and Test set
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'], axis=1), df['label'], test_size=0.2, random_state=0)

# =============================================================================
# Without data split
# =============================================================================
trainX = df.drop(['label'], axis=1)
trainY = df['label']
testX = testDF



# =============================================================================
# Apply standard scaler to output from resnet50
# =============================================================================
ss = StandardScaler()
ss.fit(X_train)
Xtrain = ss.transform(X_train)
Xtest = ss.transform(X_test)

# =============================================================================
# Take PCA to reduce feature space dimensionality
# =============================================================================
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
Xtrain = pca.transform(Xtrain)
Xtest = pca.transform(Xtest)

# =============================================================================
# Classification models on split data after PCA Transformation
# =============================================================================
# =============================================================================
# for model in range(5):
#     if(model == 0):
#         print("Accuracy of Logistic_Regression: ",  Logistic_Regression(trainX, trainY, y_train, y_test))
#     elif(model == 1):
#         print("Accuracy of SVM: ", SVM(Xtrain, Xtest, y_train, y_test))
#     elif(model == 2):
#         print("Accuracy of K_Nearest_Neighbors: ", K_Nearest_Neighbors(Xtrain, Xtest, y_train, y_test))
#     elif(model == 3):
#         print("Accuracy of Naive_Bayes: ", Naive_Bayes(Xtrain, Xtest, y_train, y_test))
#     elif(model == 4):
#         print("Accuracy of Decision_Tree: ", Decision_Tree(Xtrain, Xtest, y_train, y_test))
#     elif(model == 5):
#         print("Accuracy of Random_Forest: ", Random_Forest(Xtrain, Xtest, y_train, y_test))
# =============================================================================
        
        
# =============================================================================
# Classification models on split data without PCA Transformation
# =============================================================================
# =============================================================================
# for model in range(5):
#     if(model == 0):
#         print("Accuracy of Logistic_Regression: ",  Logistic_Regression(Xtrain, Xtest, y_train, y_test))
#     elif(model == 1):
#         print("Accuracy of SVM: ", SVM(X_train, X_test, y_train, y_test))
#     elif(model == 2):
#         print("Accuracy of K_Nearest_Neighbors: ", K_Nearest_Neighbors(X_train, X_test, y_train, y_test))
#     elif(model == 3):
#         print("Accuracy of Naive_Bayes: ", Naive_Bayes(X_train, X_test, y_train, y_test))
#     elif(model == 4):
#         print("Accuracy of Decision_Tree: ", Decision_Tree(X_train, X_test, y_train, y_test))
#     elif(model == 5):
#         print("Accuracy of Random_Forest: ", Random_Forest(X_train, X_test, y_train, y_test))
# =============================================================================
        
        
# =============================================================================
        
# =============================================================================
# using sequential Neural Network
# =============================================================================
classifierAN_PCA = Sequential()
classifierAN_PCA.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 512))
classifierAN_PCA.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifierAN_PCA.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
classifierAN_PCA.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
classifierAN_PCA.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
classifierAN_PCA.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifierAN_PCA.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifierAN_PCA.fit(Xtrain, y_train, batch_size = 10, nb_epoch = 1000)
y_predAN_PCA = classifierAN_PCA.predict(Xtest)

print(accuracy_score(y_test,y_predAN_PCA)) 

listSample = []
listSample = list(y_predAN_PCA)

for i,s in enumerate(listSample):
    if s < 0.5:
        listSample[i] = 0
    else:
        listSample[i] = 1
        
print("Accuracy: ", accuracy_score(y_test,listSample))
print("Precision: ", precision_score(y_test,listSample))
print("recall: ", recall_score(y_test,listSample))
print("F1 Score: ", f1_score(y_test,listSample))
print("Confusion Matrix: ", confusion_matrix(y_test,listSample))


# =============================================================================
# using sequential Neural Network
# =============================================================================
# =============================================================================
# classifierAN = Sequential()
# classifierAN.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu', input_dim = 2048))
# classifierAN.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu', input_dim = 1024))
# classifierAN.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 512))
# classifierAN.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# classifierAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classifierAN.fit(X_train, y_train, batch_size = 10, nb_epoch = 1000)
# y_predAN = classifierAN.predict(X_test)
# =============================================================================

# =============================================================================
# 
# # =============================================================================
# # classification models with grid search and cross validation# 
# # =============================================================================
# 
# # Logistic Regression
# 
# # Create regularization penalty space
# penalty = ['l1', 'l2']
# # Create regularization hyperparameter space
# C = np.logspace(0, 4, 10)
# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)
# from sklearn import linear_model
# # Create logistic regression
# logistic = linear_model.LogisticRegression()
# # Create grid search using 5-fold cross validation
# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, n_jobs=-1)
# # best model
# best_model = clf.fit(Xtrain, y_train)
# # View best hyperparameters
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])
# # prediction on test data using the best model 
# y_pred_best_logistic = best_model.predict(X_test)
# 
# print(accuracy_score(y_test,y_pred_best_logistic))
# 
# # =============================================================================
# # SVM
# from sklearn import svm
# 
# parameter_candidates = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
# ]
# 
# # Create a classifier object with the classifier and parameter candidates
# clf_svm = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
# # Train the classifier on data1's feature and target data
# best_model_svmGrid = clf_svm.fit(Xtrain, y_train) 
# # prediction on test data using the best model 
# y_pred_svmGrid= best_model_svmGrid.predict(Xtest)
# # get accuracy
# print(accuracy_score(y_test,y_pred_svmGrid))
# 
# # =============================================================================
# # Random Forest
# 
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1)
# model_randomForest = clf.fit(X_train, y_train)
# # Make new observation
# observation = [[ 5,  4,  3,  2]]              
# # Predict observation's class    
# y_pred_randomForest = model_randomForest.predict(X_test)
# 
# print(accuracy_score(y_test,y_pred_randomForest))
# # =============================================================================
# 
# =============================================================================


# 
# # Neural Network:
# 
# 
# classifierAN = Sequential()
# classifierAN.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 512))
# classifierAN.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# classifierAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classifierAN.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# y_predAN = classifierAN.predict(X_test)
# 
# print(confusion_matrix(y_test,y_predAN))
# print(accuracy_score(y_test,y_predAN))
# print(precision_score(y_test, y_predAN))
# print(recall_score(y_test,y_predAN))
# print(f1_score(y_test,y_predAN))
# 
# 
# classifierAN = Sequential()
# classifierAN.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu', input_dim = 2048))
# classifierAN.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
# classifierAN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# classifierAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classifierAN.fit(X_train, y_train, batch_size = 10, nb_epoch = 600)
# y_predAN = classifierAN.predict(X_test)
# 
# print(confusion_matrix(y_test,y_predAN))
# print(accuracy_score(y_test,y_predAN))
# print(precision_score(y_test, y_predAN))
# print(recall_score(y_test,y_predAN))
# print(f1_score(y_test,y_predAN))
# 
# =============================================================================

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
# 
# =============================================================================


# =============================================================================
# Best Penalty: l2
# Best C: 166.81005372000593
# =============================================================================



