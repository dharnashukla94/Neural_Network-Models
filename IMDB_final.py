#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:30:43 2019

@author: nammu
"""

#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import urllib
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

from gensim.utils import simple_preprocess
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.utils import to_categorical
from tf.keras.models import Sequential
from tf.keras.layers import Embedding, Dense, Dropout, SimpleRNN, LSTM  
from tf.keras import optimizers

#%%
def load_data(paths):
    texts = []
    reviews = []
    
    for path in paths:
        for file in os.listdir(path):
            # get review
            review = file.split('_')[1]
            review = review.split('.')[0]
            file = os.path.join(path, file)
            with open(file, "r", encoding='utf-8') as f:
                text = []
                for line in f:
                    text += simple_preprocess(line)
                texts.append(text)
                reviews.append(review)
        
    return [texts, reviews]

#%%
IMDB_directory = "./aclImdb/"

train_data, train_labels = load_data([IMDB_directory+"train/neg/", IMDB_directory+"train/pos/"])
test_data, test_labels = load_data([IMDB_directory + "test/neg/", IMDB_directory+"test/pos/"])

#%%
X = list(train_data + test_data)
y = list(train_labels + test_labels)
# Convert reviews to positive and negative
#negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10
y = [int(i)>= 7 for i in y]

#%% Glove implementation of dimension 50

EMBEDDING_DIMENSION = 50 
data_directory = './data/glove'

if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

glove_weights_file_path = os.path.join(data_directory, f'glove.6B.{EMBEDDING_DIMENSION}d.txt')

if not os.path.isfile(glove_weights_file_path):
    glove_fallback_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    local_zip_file_path = os.path.join(data_directory, os.path.basename(glove_fallback_url))
    
    if not os.path.isfile(local_zip_file_path):
        print(f'Retreiving glove weights from {glove_fallback_url}')
        urllib.request.urlretrieve(glove_fallback_url, local_zip_file_path)
               
    with zipfile.ZipFile(local_zip_file_path, 'r') as z:
        print(f'Extracting glove weights from {local_zip_file_path}')
        z.extractall(path=data_directory)
        
        
#%%
embed_ind = {}
glove_filename = './glove.6B/glove.6B.50d.txt'

with open(glove_filename, "r", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeff = np.asarray(values[1:], dtype='float32')
        embed_ind[word] = coeff

print('Contains %s word vectors.' % len(embed_ind))
        
#%%
MAX_SEQUENCE_LENGTH=500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(y))


#%%

embed_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIMENSION))
for word, i in word_index.items():
    embed_vector = embed_ind.get(word)
    if embed_vector is not None:
        embed_matrix[i] = embed_vector


#%%
# Divide the training testing set. 25k each. 
X_train, y_train = data[:25000], labels[:25000]
X_test, y_test = data[25000:], labels[25000:]

#%% Implementation of classification models

# Vanilla RNN


def vanilla_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
    model = Sequential()
    model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False, weights=[embed_matrix]))
    model.add(SimpleRNN(units=state, input_shape=(num_words,1), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='sigmoid'))
    
    RMS = optimizers.RMSprop(lr = lra)
    model.compile(loss = 'binary_crossentropy', optimizer = RMS, metrics = ['accuracy'])
    
    return model


# LSTM
def lstm_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
    model = Sequential()
    model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False, weights=[embed_matrix]))
    model.add(LSTM(state))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='sigmoid'))
    RMS = optimizers.RMSprop(lr = lra)
    model.compile(loss='binary_crossentropy', optimizer=RMS, metrics=['accuracy'])
    
    return model

#%%
# Run the models

def run_model(state, lr, batch, dropout, model, epoch=5, num_outputs=2, emb_dim=100, input_length=2380):
        
    num_words = len(word_index)
    if model == "lstm": 
        model = lstm_rnn(num_words, state, lr, dropout)
    elif model == "vanilla":
        model = vanilla_rnn(num_words, state, lr, dropout)
        epoch = 10
        
    #model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=1)

    testscore = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', testscore[0])
    print('Test accuracy:', testscore[1])
    
    return [history.history, testscore]

#%%

def hypruns(state, comb, model):
    history = []
    testscore = []
    itera = 1
    for i in range(itera):
        l, b, d = comb
        print("state %s, lr %s, batch %s, dropout %s." %(state, l, b, d))
        res = run_model(state, l, b, d, model)
        
        if res:
            history.append(res[0])
            testscore.append(res[1])
    
    # take avg of testscore
    testscore = list(np.mean(np.array(testscore), axis=0))
    hyps = [state] + comb
    
    return [history, testscore, hyps]

#%%
# Tuning the hyper-parameters
def tunehyps(states, comb, model):
    res = []
    hist = []
#    for state in states:
    for comb in combs:
        history, testscore, hyps = hypruns(state, comb, model)
        res.append(testscore + hyps)
        hist.append(history)
                
#    if not os.path.isdir('./results/'+model+'/testscore_'+'state_'+str(state)):
        os.makedirs('./results/'+model+'/testscore_'+'state_'+str(state))    
    
    # save testscore to file
    with open('./results/'+model+'/testscore_'+'state_'+str(state), 'w', encoding="utf-8") as fout:
        pprint(res, fout)

#    if not os.path.isdir('./results/'+model+'/history_'+'state_'+str(state)):
        os.makedirs('./results/'+model+'/history_'+'state_'+str(state))  

    # save history to file
    with open('./results/'+model+'/history_'+'state_'+str(state), 'w', encoding="utf-8") as fout:
        pprint(hist, fout)

#%%
## Test for various cases of hyper-parameters
#
#states = [20, 50, 100, 200, 500]
#learn_rate = [ 0.01, 0.1]
#batch = [100, 300]
#dropout = [0.1, 0.5]
#repeats = 1
#model = ["lstm", "vanilla"]
#model = model[0]
#
#
#for combs in itertools.product([ 0.01, 0.1], [100, 300],[0.1, 0.5] ):
#    print(combs)
#


#%%
state =200

combs = [[0.01, 300, 0.5]]
model = "lstm"

tunehyps(state, combs,  model)

