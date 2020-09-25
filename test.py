import csv
import os
import numpy as np
import random
import re
import constant
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from nltk.stem import SnowballStemmer
from array import array
import dask.array as da
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

vocabulary = {}
stemmer = SnowballStemmer('russian')
maketrans = str.maketrans
num_words = 130000
filterStr = r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
split=" "
word_count = 200

def get_data() -> list:
    label = 0 
    labels = list()
    messages = list()
    messageGroups = list()
    for names in os.listdir(constant.PATH) :
        with open(constant.PATH + names, newline = '', encoding = constant.ENCODING) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            messageGroup = list(np.array(list(reader))[:,4])
            messageGroups.append(messageGroup)

    for messageGroup in messageGroups :
        for message in messageGroup :
            messages.append(str(message))
            labels.append(label)
        label += 1
    return messages, labels

def texts_to_sequences(messages, word_index) : 
    for message in messages :
        message = message.lower()
        translate_map = maketrans(filterStr, split * len(filterStr))
        message = message.translate(translate_map)
        seq = message.split(split)
        vect = []
        for word in seq :
            index = word_index.get(word)
            if index is not None and num_words and index < num_words and len(vect) < word_count:
                    vect.append(index)
        yield vect

def texts_to_sequences2(message, word_index) : 
    message = message.lower()
    translate_map = maketrans(filterStr, split * len(filterStr))
    message = message.translate(translate_map)
    seq = message.split(split)
    vect = np.zeros(word_count, int)
    i = 0
    for word in seq :
        index = word_index.get(word)
        if index is not None and num_words and index < num_words and i < len(vect):
                vect[i] = index
                i += 1
    return vect

def get_bows(messages) -> list:
    bows = np.zeros(shape=(13170, word_count), dtype=int)
    sorted_voc = []
    word_counts = OrderedDict()
    for message in messages :
        message = message.lower()
        translate_map = maketrans(filterStr, split * len(filterStr))
        message = message.translate(translate_map)
        seq = message.split(split)
        for word in seq :
            if word in word_counts :
                word_counts[word] += 1
            else :
                word_counts[word] = 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)    
    sorted_voc.extend(wc[0] for wc in wcounts)
    word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
    index_word = {c: w for w, c in word_index.items()}
    vocab_size = len(word_index) + 1

    i = 0
    for message in messages :
        vect = texts_to_sequences2(message, word_index)
        if(i >= 13170) :
            break    
        bows[i] = vect
        i += 1
    return bows , vocab_size

def get_bows2(messages) -> list:
    bows = list()
    sorted_voc = []
    word_counts = OrderedDict()
    for message in messages :
        message = message.lower()
        translate_map = maketrans(filterStr, split * len(filterStr))
        message = message.translate(translate_map)
        seq = message.split(split)
        for word in seq :
            if word in word_counts :
                word_counts[word] += 1
            else :
                word_counts[word] = 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)    
    sorted_voc.extend(wc[0] for wc in wcounts)
    word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
    index_word = {c: w for w, c in word_index.items()}
    vocab_size = len(word_index) + 1

    return list(texts_to_sequences(messages, word_index)) , vocab_size


(messages, labels) = get_data()
(bows, vocab_size) = get_bows(messages)
tokenizer = Tokenizer(num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(messages)
X = tokenizer.texts_to_sequences(messages)
X_train, X_test, y_train, y_test = train_test_split(bows, labels, test_size=0.05, random_state=0)


rc = RidgeClassifier(tol=1e-2, solver="sag")
rc.fit(X_train, y_train)

y_pred = rc.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)
print('end')


test_size = len(messageModels) // 5 
(train_data, test_Data, train_label, test_label) = train_test_split(bows, labels, test_size = test_size)


