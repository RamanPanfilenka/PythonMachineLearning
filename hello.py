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

vocabulary = {}
stemmer = SnowballStemmer('russian')

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

def get_bows(messages) -> list:
    index = 0
    bows = list()
    for message in messages :
        words = re.findall(r'\w+[\-]?\w+', str(message))
        bow = np.zeros(140000, int)
        for word in words :
            stemmered_word = stemmer.stem(word)
            if not vocabulary.keys().__contains__(stemmered_word):
                vocabulary[stemmered_word] = index
                bow[index] = str(message).count(word)
                index += 1
            else :
                value = vocabulary.get(stemmered_word)
                bow[value] = str(message).count(word)
        bows.append(bow)
    
    return da.from_array(bows, chunks=(10000, 10000))




(messages, labels) = get_data()
bows = get_bows(messages)
del vocabulary
X_train, X_test, y_train, y_test = train_test_split(bows, labels, test_size=0.2, random_state=0)
rc = RidgeClassifier()
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=True, random_state=None, solver='auto',
                tol=0.001)
rc.fit(X_train, y_train)

y_pred = rc.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

print('end')


