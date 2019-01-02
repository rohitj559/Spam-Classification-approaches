# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 19:27:16 2018

@author: Rohit
"""

# more common imports
import pandas as pd
import numpy as np
from collections import Counter
import re

# languange processing imports
import nltk
from gensim.corpora import Dictionary
# preprocessing imports
from sklearn.preprocessing import LabelEncoder

# model imports
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV

# visualization imports
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
sns.set()  # defines the style of the plots to be seaborn style

import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# For Training
ham_path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\ham'
spam_path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\spam'
ham_folder = os.listdir(ham_path)
spam_folder = os.listdir(spam_path)

# For Testing  
test_Path = 'C:\\Masters-LUC\\Fall_2018\\Research\\Exercise8_Spam_Play_with_text\\text-convert-mails\\test' 
test_folder = os.listdir(test_Path)

def ListConversion(source, path):  
    mail_list = []
    for file in source:
        sourceFile = os.path.join(path, file)
        fin = open(sourceFile, "r", encoding="utf-8")
        fin.seek(0,0)
        fin = ''.join(x for x in fin)
        fin = fin.lower()
        mail_list.append(fin)
    return mail_list

# =============================================================================
# conversion of all mails into seperate dataframes
# =============================================================================
ham_DataFrame = pd.DataFrame({'text':ListConversion(ham_folder, ham_path)})
spam_DataFrame = pd.DataFrame({'text':ListConversion(spam_folder, spam_path)})
test_DataFrame = pd.DataFrame({'text':ListConversion(test_folder, test_Path)})

# =============================================================================
# assigning labels to dataframes
# =============================================================================
ham_DataFrame["label"] = [0]*len(ham_DataFrame)
spam_DataFrame["label"] = [1]*len(spam_DataFrame)

# =============================================================================
# shuffling
# =============================================================================
frames = [ham_DataFrame, spam_DataFrame]
dataframe = pd.concat(frames)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# =============================================================================
# Inspect text variable
# =============================================================================
document_lengths = np.array(list(map(len, dataframe.text.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))

# =============================================================================
# find and remove non-ascii words
# =============================================================================
# special word in a variable for later use
our_special_word = 'qwerty'

def remove_ascii_words(df):    
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'text'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                # Replacing non-ascii words with special word 'querty'
                df.loc[i, 'text'] = df.loc[i, 'text'].replace(word, our_special_word)
    return non_ascii_words

non_ascii_words = remove_ascii_words(dataframe)
print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(
    len(non_ascii_words)))


def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

# =============================================================================
# transform the documents into sentences for the word2vecmodel
# =============================================================================
def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['text'] = df.text.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists
        
        
w2v_preprocessing(dataframe)


sentences = []
for sentence_group in dataframe.tokenized_sentences:
    sentences.extend(sentence_group)

print("Number of sentences: {}.".format(len(sentences)))
print("Number of texts: {}.".format(len(dataframe)))

# Set values for various parameters
num_features = 200    # Word vector dimensionality
min_word_count = 3    # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
W2Vmodel = Word2Vec(sentences=sentences,
                    sg=1,
                    hs=0,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    negative=5,
                    iter=6)


def get_w2v_features(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

dataframe['w2v_features'] = list(map(lambda sen_group:
                                      get_w2v_features(W2Vmodel, sen_group),
                                      dataframe.tokenized_sentences))
    
# =============================================================================
# HAM_w2v_distribution = dataframe.loc[dataframe.label == 0, 'w2v_features'].mean()
# SPAM_w2v_distribution = dataframe.loc[dataframe.label == 1, 'w2v_features'].mean()
# 
# fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(20,10))
# nr_top_bars = 5
# ax1.set_title("Ham w2v feature distributions", fontsize=16)
# ax2.set_title("Spam w2v feature distributions", fontsize=16)
# ax3.text(-10, 2.3, "Average feature vectors", fontsize=30, ha="center", va="center", rotation="vertical")
# 
# for ax, distribution, color in zip([ax1,ax2], [HAM_w2v_distribution,SPAM_w2v_distribution], ['b','r']):
#     # Individual distribution barplots
#     ax.bar(range(len(distribution)), distribution, alpha=0.7)
#     rects = ax.patches
#     for i in np.argsort(distribution)[-nr_top_bars:]:
#         rects[i].set_color(color)
#         rects[i].set_alpha(1)
#     # General plotting adjustments
#     ax.set_xlim(-1, 200)
#     ax.set_xticks(range(20,199,20))
#     ax.set_xticklabels(range(20,199,20), fontsize=16)
#     ax.set_ylim(-0.8,0.8)
# 
# fig.tight_layout(h_pad=3.)
# =============================================================================
    
def get_cross_validated_model(model, param_grid, X, y, nr_folds=5):
    """ Trains a model by doing a grid search combined with cross validation.
    args:
        model: your model
        param_grid: dict of parameter values for the grid search
    returns:
        Model trained on entire dataset with hyperparameters chosen from best results in the grid search.
    """
    # train the model (since the evaluation is based on the logloss, we'll use neg_log_loss here)
    grid_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_log_loss', cv=nr_folds, n_jobs=-1, verbose=True)
    best_model = grid_cv.fit(X, y)
    # show top models with parameter values
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    display(result_df[show_columns].sort_values(by='rank_test_score').head())
    return best_model

# since train_data['lda_features'] and train_data['w2v_features'] don't have the needed shape and type yet,
# we first have to transform every entry
# X_train_lda = np.array(list(map(np.array, train_data.lda_features)))    
X_train_w2v = np.array(list(map(np.array, dataframe.w2v_features)))
X_w2v = np.array(list(map(np.array, dataframe.w2v_features)))
# X_train_combined = np.append(X_train_lda, X_train_w2v, axis=1)

# store all models in a dictionary
models = dict()

# Word2Vec features only
lr = LogisticRegression()
param_grid = {'penalty': ['l1', 'l2']}
best_lr_w2v = get_cross_validated_model(lr, param_grid, X_train_w2v, dataframe.label)

models['best_lr_w2v'] = best_lr_w2v


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_w2v, dataframe.label, test_size=0.2, random_state=0)

# =============================================================================
# Logistic Regression
# =============================================================================

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
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, n_jobs=-1)
# best model
best_model = clf.fit(X_train, y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# prediction on test data using the best model 
y_pred_best_logistic = best_model.predict(X_test)

print(accuracy_score(y_test,y_pred_best_logistic))
print("Accuracy: ", accuracy_score(y_test,y_pred_best_logistic))
print("Precision: ", precision_score(y_test,y_pred_best_logistic))
print("recall: ", recall_score(y_test,y_pred_best_logistic))
print("F1 Score: ", f1_score(y_test,y_pred_best_logistic))
print("Confusion Matrix: ", confusion_matrix(y_test,y_pred_best_logistic))


# =============================================================================
# svm
# =============================================================================

from sklearn import svm

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier object with the classifier and parameter candidates
clf_svm = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
# Train the classifier on data1's feature and target data
best_model_svmGrid = clf_svm.fit(X_train, y_train) 
# prediction on test data using the best model 
y_pred_svmGrid= best_model_svmGrid.predict(X_test)
# get accuracy
print(accuracy_score(y_test,y_pred_svmGrid))
print("Accuracy: ", accuracy_score(y_test,y_pred_svmGrid))
print("Precision: ", precision_score(y_test,y_pred_svmGrid))
print("recall: ", recall_score(y_test,y_pred_svmGrid))
print("F1 Score: ", f1_score(y_test,y_pred_svmGrid))
print("Confusion Matrix: ", confusion_matrix(y_test,y_pred_svmGrid))


# =============================================================================
# # Mutinomial Naive Bayes
# =============================================================================
prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)










