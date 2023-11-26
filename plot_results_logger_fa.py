#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:27:12 2022

@author: ifeanyi.ezukwoke
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 124
import os
import random
import gensim
import numpy as np
import pandas as pd
import multiprocessing
from os.path import join
import tensorflow as tf
from keras.models import load_model
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
from utils import plot_latent_space, compute_metric, model_saver
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 100

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Import data

datatype = "fa"
batch_size = 64
VOCAB_SIZE = 1000
#import data
# path = os.getcwd()


x_n_pp = pd.read_csv(join(path, "x_n/x_n_pp.csv"), sep = ';')[x_n]
path_xn_pp = pd.read_csv(join(path, "path_data/path_n_pp.csv"), sep = ';', low_memory = False)
tx = pd.concat([x_n_pp, path_xn_pp], axis=1)
txy = tx.astype(str).apply(lambda x: Mergefeatures(x).concat(), axis = 1)


#train VAE model using cosine similarity
model = gensim.models.Word2Vec.load(join(path, f"Model/word2vec_{VOCAB_SIZE}.model"))
vectorize_wv = SimilarityMat(model, txy).wordEmbed()
#save vectorized datat set
#np.save(os.path.join(path, "Model/x_vector.npy"), vectorize_wv)
#------
n, m = vectorize_wv.shape
x_train, x_test = train_test_split(vectorize_wv, test_size = 0.30)
vectorize_wv = vectorize_wv.reshape(vectorize_wv.shape[0], vectorize_wv.shape[1], 1)
x_train_r = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_r = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train_r)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(x_test_r)
test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(batch_size)


inp_dim = x_train_r.shape[1:]

#%% for Zhiqiang

# path  = '/home/ifeanyi.ezukwoke/Documents/FA4.0/Zhiqiang'
# files_npy = [x for x in os.listdir(path) if '.npy' in x]
# dt = {f"{x.split('.')[0]}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
#       in zip(files_npy, files_npy)}


# fig, ax = plt.subplots(1, 6)
# fig.subplots_adjust(hspace = .5, wspace = .001)
# ax = ax.ravel()
        
# for w, (i, j) in zip(range(6), dt.items()):
#     for q, (n, m) in zip(range(6),  j.items()):
#         ax[q].plot(range(20), m, label = f"{i.split('_')[1].upper()}", lw=.7)
#         ax[q].set_xlabel('epochs')
#         ax[q].set_ylabel(f'{n}')
#         ax[q].set_title(f'{n.upper()}')
#         ax[q].legend()
#     # ax[w].set_title(f'{i}')

#%% For Kenneth...Visualize metrics..
datatype = "fa"
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 200
hidden_dim = 512
latent_dims = 2
archi_type = 'v1'
#params
distrib_type = 'g'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
save_model_after = 2
    
paths  = '/home/ifeanyi.ezukwoke/Documents/FA4.0/kenneth/Scripts/Modeling/Autoencoder'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(paths, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(paths, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
dt = {f"{x.upper()}": np.load(os.path.join(paths, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}
    
fig, ax = plt.subplots(1, len(logger))
fig.subplots_adjust(hspace = .5, wspace = .001)
ax = ax.ravel()
        
for w, (i, j) in zip(range(len(logger)), logger.items()):
    for q, (n, m) in zip(range(7),  j.items()):
        ax[q].plot(range(epochs), m, label = f"{i}", lw=.7)
        ax[q].set_title(f'{n.upper()}')
        ax[q].set_xlabel('epochs')
        ax[q].set_ylabel(f'{n}')
        ax[q].legend()
    # ax[w].set_title(f'{i}')
        
#%%
logger = np.load('/home/kenneth/Documents/FA4.0/kenneth/TechInterviews/REMED/g/gcvae/med/latent_2/200/mmd/loggers.npy', allow_pickle=True).ravel()[0]
z = np.load('/home/kenneth/Documents/FA4.0/kenneth/TechInterviews/REMED/g/gcvae/med/latent_2/200/mmd/results.npy', allow_pickle=True).ravel()[0]
fig, ax = plt.subplots(1, len(logger))
fig.subplots_adjust(hspace = .5, wspace = .001)
ax = ax.ravel()

for w, (i, j) in enumerate(zip(logger.keys(), logger.values())):
    ax[w].plot(range(epochs), j, label = f"{i}", lw=.7)
    ax[w].set_title(f'{i.upper()}')
    ax[w].set_xlabel('epochs')
    ax[w].set_ylabel(f'{i}')
    ax[w].legend()
    
#%% For kenneth...Latent space...

datatype = "fa"
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 200
hidden_dim = 512
latent_dims = 2
archi_type = 'v1'
#params
distrib_type = 'g'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
save_model_after = 2
    
path  = f'/home/ifeanyi.ezukwoke/Documents/FA4.0/kenneth/Scripts/Modeling/Autoencoder/{distrib_type}'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/model.h5"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/model.h5"))
            
dt = {f"{x.upper()}": load_model(os.path.join(path, f'{y}')) for (x, y)\
      in zip(lls, file_path)}
    
# plot_latent_space(model, n= 10)

fig, ax = plt.subplots(1, len(dt))
fig.subplots_adjust(hspace = .5, wspace = .001)
ax = ax.ravel()


for w, (i, j) in zip(range(len(dt)), dt.items()):
    mu, sigma, z = j.encoder.predict(x_test, batch_size = batch_size)
    ax[w].scatter(z[:, 0], z[:, 1], label = f"{i}", s=.5)
    ax[w].set_title(f'{i.upper()}')
    ax[w].set_xlabel('z[0]')
    ax[w].set_ylabel('z[1]')
    ax[w].legend()

#%% for Kenneth...plot 2-D img representation...
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 150

epochs = 200
latent_dims = 2
path  = '/home/ifeanyi.ezukwoke/Documents/FA4.0/kenneth/Scripts/Modeling/Autoencoder'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/model.h5"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/model.h5"))
            
dt = {f"{x.upper()}": load_model(os.path.join(path, f'{y}')) for (x, y)\
      in zip(lls, file_path)}
    
    
fig, ax = plt.subplots(1, len(dt))
fig.subplots_adjust(hspace = .5, wspace = .001)
ax = ax.ravel()
        
for w, (p, q) in zip(range(len(dt)), dt.items()):
    # mu, sigma, z = j.encoder.predict(x_test, batch_size = batch_size)
    n = 10
    digit_size = 28
    scale = 1.0
    figsize = 10
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = q.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    
    #plt.figure(figsize=(figsize, figsize))
    ax[w].axis('off')
    ax[w].set_xlabel(f'({chr(w+97)}) {p}')
    ax[w].imshow(figure, cmap="Greys_r")
    ax[w].set_title(f'({chr(w+97)}) {p}',y=-0.1,pad=-14)
    




#%% Disentanglement Metric....| results.npy

epochs = 200
latent_dims = 2
path  = f'/home/ifeanyi.ezukwoke/Documents/FA4.0/kenneth/Scripts/Modeling/Autoencoder/{distrib_type}'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/results.npy"))
            
metrics = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}


print('-'*120)
print("|\t\t Model \t\t|\t\t Factor-VAE \t\t|\t\t MIG \t\t|\t\t Modularity \t\t|\t\t Jemmig \t\t|")
print('-'*120)
for i, j in metrics.items():
    print(f"|\t {i} \t\t|\t\t {j['factorvae_score_mu']:.3f} +/- {j['factorvae_score_sigma']:.3f} \t\t|"+
          f"\t\t {j['mig_score_mu']:.4f} \t\t|\t\t {j['modularity']:.3f} \t\t|\t\t {j['jemmig']:.3f} \t\t|")
print('-'*120)

#%% Total losses, Reconstruction and KL-divergence...


epochs = 200
latent_dims = 2
path  = f'/home/ifeanyi.ezukwoke/Documents/FA4.0/kenneth/Scripts/Modeling/Autoencoder/{distrib_type}'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
logger = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}


print('-'*120)
print("|\t\t Model \t\t|\t\t Total loss \t\t|\t\t Reconstruction \t\t|\t\t KL divergence |")
print('-'*120)
for i, j in logger.items():
    print(f"|\t {i} \t\t\t|\t\t\t {j['elbo'][-1]:.3f} \t\t\t|\t\t\t {j['reconstruction'][-1]:.3f} \t\t\t|"+
          f"\t\t\t {j['kl_div'][-1]:.4f} \t\t\t|")
print('-'*120)

    
    
    
    
    
    


