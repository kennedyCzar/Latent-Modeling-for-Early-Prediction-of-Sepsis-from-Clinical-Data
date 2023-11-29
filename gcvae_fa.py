#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:06:29 2022

@author: ifeanyi.ezukwoke
"""

import os
import gensim
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import multiprocessing
from os.path import join
import tensorflow as tf
from train_gcvae import train_gcvae as gcvae
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.patheffects as pe
tf.config.run_functions_eagerly(True)
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from utils import (plot_latent_space, compute_metric, 
                   model_saver, model_saver_scriterion)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 120


#%% Import data

datatype = "med"
batch_size = 120
#import data
path = '/home/ifezukwo/REMEDS'
#path = '/home/kenneth/Documents/FA4.0/kenneth/TechInterviews/REMED'

#%% Comment out this section is data.npz in already existing...

# #load the training data
# train = pd.read_csv(join(path, 'train.csv')).iloc[:, 1:]
    
# #define the validation set as 15% of the training set
# N = len(train)
# train = train.sample(frac=1).reset_index(drop=True)
# val = train[:int(N*0.2)]
# train = train[int(N*0.2):]

# #separate out the training data
# #note that input data for LSTM should be in the format of (number_samples, sequence_len, num_channels)
# y_train = train['label']
# y_train[y_train.isna()] = np.pi
# y_train = np.asarray(list(y_train)) 

# X_train_cont = np.asarray(list(train['X_cont'].values))
# X_train_cat = train.drop(['X_cont', 'label'], axis=1)

# #for the mask layer, any NaN values need to be replaced by a unique value. use the constant np.pi
# X_train_cat[X_train_cat.isna()] = np.pi
# X_train_cat = np.asarray(list(X_train_cat.values))

# #load the test data
# test = pd.read_csv(join(path, 'test.csv')).iloc[:, 1:]

    
# #y_test = np.asarray(list(test['label']))
# y_test = test['label']
# y_test[y_test.isna()] = np.pi
# y_test = np.asarray(list(y_test))
# X_test_cont = np.asarray(list(test['X_cont'].values))
# X_test_cat = test.drop(['X_cont', 'label'], axis=1)

# #replace NaN with pi
# X_test_cat[X_test_cat.isna()] = np.pi
# X_test_cat = np.asarray(list(X_test_cat.values))

#%%
from sklearn.model_selection import train_test_split
# X_train_cat, _, y_train, _ = train_test_split(X_train_cat, y_train, test_size = 0.80, random_state = 42)
# X_test_cat, _, y_test, _ = train_test_split(X_test_cat, y_test, test_size = 0.50, random_state = 42)
# X_train_cont, _, = train_test_split(X_train_cont, test_size = 0.80, random_state = 42)
# X_test_cont, _, = train_test_split(X_test_cont, test_size = 0.50, random_state = 42)

# np.savez_compressed(join(path, 'data'), X_train_cat = X_train_cat,
#                     y_train = y_train,
#                     y_test = y_test,
#                     X_test_cat = X_test_cat,
#                     X_train_cont = X_train_cont,
#                     X_test_cont = X_test_cont
#                     )

tt = np.load(join(path, 'data.npz'))
X_train_cat, y_train, X_test_cat, y_test, X_train_cont, X_test_cont = tt['X_train_cat'],\
                                                                        tt['y_train'],\
                                                                            tt['X_test_cat'],\
                                                                                tt['y_test'],\
                                                                                    tt['X_train_cont'],\
                                                                                        tt['X_test_cont']
#reduced dataset for test on mini-PC
# X_train_cat = X_train_cat[:100]
# X_test_cat = X_test_cat[:100]

n, m = X_train_cat.shape
x_train_r = X_train_cat.reshape(X_train_cat.shape[0], X_train_cat.shape[1], 1)
x_test_r = X_test_cat.reshape(X_test_cat.shape[0], X_test_cat.shape[1], 1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train_r)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(x_test_r)
test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(batch_size)


inp_dim = x_train_r.shape[1:]


#%% Modeling
loss_index = 3
# vae_type = 'gcvae' #else infovae
inp_shape =  x_train_r.shape[1:]
num_features = inp_shape[0]

#the parameters are only to change fixed weights
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 10), 0),
            #'controlvae': (0, 0), # No need for this since it is embedded in GCVAE
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }


for lat in [3, 10, 15]:
    lr = 1e-3
    epochs = 10
    hidden_dim = 512
    latent_dims = lat
    loss_type = list(params.keys())[loss_index] #elbo -> 0; betavae -> 1; controlvae -> 2; infovae -> 3; gcvae -> 4
    archi_type = 'v1'
    #params
    distrib_type = 'g'
    beta, gamma = params[f'{loss_type}']
    mmd_typ = 'mmd' #['mmd', 'mah', 'mah_rkhs', 'mah_gcvae']
    save_model_arg = False
    save_model_after = 50
    #-----------------------------ignore this section if no stopping criterion is needed--------------------------------------
    stop_criterion = 'noStop' #'useStop' or 'igstop'  --> useStop --> Use stopping criterion or igstop --> Ignore stopping
    stopping = True if stop_criterion == 'useStop' else False
    pid_a = True if stopping == True else False
    pid_b = True if stopping == True else False
    #-----------------------------------------end stopping criterion params --------------------------------------------------
    model = gcvae(inp_shape = inp_shape, 
                        num_features = num_features, 
                        hidden_dim = hidden_dim, 
                        latent_dim = latent_dims, 
                        batch_size = batch_size, 
                        beta = beta,
                        gamma = gamma,
                        dist = distrib_type,
                        vloss = loss_type,
                        lr = lr, 
                        epochs = epochs,
                        architecture = archi_type,
                        mmd_type = mmd_typ).fit(train_dataset, X_test_cat,
                                                datatype, stopping = stopping,
                                                save_model = save_model_arg,
                                                save_model_iter = save_model_after,
                                                pid_a = pid_a,
                                                pid_b = pid_b,)
                                                    
        
    #save model....
    model_saver(model,\
                path,\
                X_train_cat,\
                X_test_cat,\
                hidden_dim,\
                latent_dims,\
                batch_size,\
                beta,\
                gamma,\
                distrib_type,\
                loss_type,\
                lr,\
                epochs,\
                archi_type,\
                mmd_typ,\
                datatype)

                
                
#%% Testing...

import pytest

@pytest.fixture
def setup_data():
    path = '/home/ifezukwo/REMEDS'
    tt = np.load(join(path, 'data.npz'))
    tt = np.load(join(path, 'data.npz'))
    X_train_cat, y_train, X_test_cat, y_test, X_train_cont, X_test_cont = tt['X_train_cat'],\
                                                                            tt['y_train'],\
                                                                                tt['X_test_cat'],\
                                                                                    tt['y_test'],\
                                                                                        tt['X_train_cont'],\
                                                                                            tt['X_test_cont']
    n, m = X_train_cat.shape
    x_train_r = X_train_cat.reshape(X_train_cat.shape[0], X_train_cat.shape[1], 1)
    x_test_r = X_test_cat.reshape(X_test_cat.shape[0], X_test_cat.shape[1], 1)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train_r)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test_r)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(32)

    inp_dim = x_train_r.shape[1:]
    
    return train_dataset, X_test_cat, inp_dim

def test_model_training(setup_data):
    train_dataset, X_test_cat, inp_dim = setup_data

    loss_index = 3
    params = {
        'elbo': (1, 0),
        'betavae': ((1, 10), 0),
        'infovae': (0, 500),
        'gcvae': (1, 1),
        }

    for lat in [3, 10, 15]:
        lr = 1e-3
        epochs = 10
        hidden_dim = 512
        latent_dims = lat
        loss_type = list(params.keys())[loss_index]
        archi_type = 'v1'
        distrib_type = 'g'
        beta, gamma = params[f'{loss_type}']
        mmd_typ = 'mmd'
        save_model_arg = False
        save_model_after = 50
        stopping = True
        pid_a = True if stopping else False
        pid_b = True if stopping else False

        model = gcvae(inp_shape=inp_dim,
                      num_features=inp_dim[0],
                      hidden_dim=hidden_dim,
                      latent_dim=latent_dims,
                      batch_size=32,
                      beta=beta,
                      gamma=gamma,
                      dist=distrib_type,
                      vloss=loss_type,
                      lr=lr,
                      epochs=epochs,
                      architecture=archi_type,
                      mmd_type=mmd_typ).fit(train_dataset, X_test_cat,
                                           datatype, stopping=stopping,
                                           save_model=save_model_arg,
                                           save_model_iter=save_model_after,
                                           pid_a=pid_a,
                                           pid_b=pid_b)

        assert model is not None


#test_model_training(setup_data)

                