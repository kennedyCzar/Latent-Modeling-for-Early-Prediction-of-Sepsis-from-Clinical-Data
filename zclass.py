#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:17:23 2023

@author: kenneth
"""

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
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 100

#path = '/home/ifezukwo/REMEDS'
path = '/home/kenneth/Documents/FA4.0/kenneth/TechInterviews/REMED'


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
# log_cols=["Classifier", "Accuracy", "Log Loss"]
# log = pd.DataFrame(columns=log_cols)
#%% Disentanglement metrics

datatype = "med"
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

#params
epochs = 10
distrib_type = 'g'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
latent_dims = 15
lls = ['VAE', r'$\beta$-VAE', 'InfoVAE', 'GCVAE']

file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/results.npy"))
            
met = {f"{x}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}
    
print('-'*120)
print("|\t\t Model \t\t|\t\t Factor-VAE \t\t|\t\t MIG \t\t|\t\t Modularity \t\t|\t\t Jemmig \t\t|")
print('-'*120)
for i, j in met.items():
    print(f"|\t {i} \t\t|\t\t {j['factorvae_score_mu']:.2f} +/- {j['factorvae_score_sigma']:.2f} \t\t|"+
          f"\t\t {j['mig_score_mu']:.2f} \t\t|\t\t {j['modularity']:.2f} \t\t|\t\t {j['jemmig']:.2f} \t\t|")
print('-'*120)

#%% Loggers metric

file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
logger = {f"{x}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}


print('-'*120)
print("|\t\t Model \t\t|\t\t Total loss \t\t|\t\t Reconstruction \t\t|\t\t KL divergence |")
print('-'*120)
for i, j in logger.items():
    print(f"|\t {i} \t\t\t|\t\t\t {j['elbo'][-1]:.3f} \t\t\t|\t\t\t {j['reconstruction'][-1]:.3f} \t\t\t|"+
          f"\t\t\t {j['kl_div'][-1]:.4f} \t\t\t|")
print('-'*120)

#%% Plot losses...

fig, ax = plt.subplots(1, len(logger))
fig.subplots_adjust(hspace = .5, wspace = .001)
ax = ax.ravel()
        
for w, (i, j) in zip(range(len(logger)), logger.items()):
    for q, (n, m) in zip(range(4),  j.items()):
        ax[q].plot(range(epochs), m, label = f"{i}", lw = 1.7, marker = '*')
        ax[q].set_title(f'{n.upper()}')
        ax[q].set_xlabel('epochs')
        ax[q].set_ylabel(f'{n}')
        ax[q].legend()
    ax[w].set_title(f'{i}')
    
#%%
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
    
    # log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    # log = log.append(log_entry)
    
print("="*30)
