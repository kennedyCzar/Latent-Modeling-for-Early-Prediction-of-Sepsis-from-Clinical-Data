#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 03:20:45 2023

@author: kenneth
"""

import os
from os.path import join
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import TimeDistributed, Bidirectional, BatchNormalization, Dropout, Input, Add, Masking
from keras import Model
import pdb
import pandas as pd
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
import glob
import sys
import pickle
#os.listdir(path)

path = '/home/ifezukwo/REMEDS'
#path = '/home/kenneth/Documents/FA4.0/kenneth/TechInterviews/REMED'

#%%

tt = np.load(join(path, 'data.npz'))
X_train_cat, y_train, X_test_cat, y_test, X_train_cont, X_test_cont = tt['X_train_cat'],\
                                                                        tt['y_train'],\
                                                                            tt['X_test_cat'],\
                                                                                tt['y_test'],\
                                                                                    tt['X_train_cont'],\
                                                                                        tt['X_test_cont']
                                                                                        
#%%

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

classifiers = [
      MLPClassifier(
      activation = 'tanh',
      solver = 'lbfgs',
      early_stopping = False,
      hidden_layer_sizes = (40,10,10,10,10, 1),
      random_state = 1,
      batch_size = 'auto',
      max_iter = 100,
      learning_rate_init = 1e-5,
      tol = 1e-4,
  ),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

x = 200
for clf in classifiers:
    clf.fit(X_train_cat, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test_cat)
    acc = accuracy_score(y_test, train_predictions)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, train_predictions)
    auc = metrics.auc(fpr, tpr)
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    
    train_predictions = clf.predict_proba(X_test_cat)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
