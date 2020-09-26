import os
import csv
import math
import nltk
import constant
import numpy as np
import collections
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stemmer = SnowballStemmer('russian')
stemmer.stopwords = stopwords.words("russian")


def get_data() -> list:
    label = 0 
    labels = list()
    messages = list()
    messageGroups = list()
    for names in os.listdir(constant.PATH) :
        with open(constant.PATH + names, newline = '', encoding = constant.ENCODING) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            messageGroup = list(np.array(list(reader)[:500])[:,4])
            messageGroups.append(messageGroup)

    for messageGroup in messageGroups :
        for message in messageGroup :
            messages.append(str(message))
            labels.append(label)
        label += 1
    return messages, labels

def get_num_of_words(tokens, uniqWords):
    num_of_words = []
    for token in tokens:
        numOfWord = dict.fromkeys(uniqWords.keys(), 0)
        for word in token:
            numOfWord[word] += 1
        num_of_words.append(numOfWord)
    
    return num_of_words

def get_tfidf(num_of_words, tokens, uniqWords):
    N = len(tokens)
    for index in range(len(num_of_words)):
        bagOfWordsCount = len(tokens[index])
        for word, count in num_of_words[index].items():
            if count > 0 :
                num_of_words[index][word] = count / float(bagOfWordsCount) * math.log(N / float(uniqWords[word]))
            else :
                num_of_words[index][word] = 0
    return num_of_words

def convert_tfids_to_data(tfids):
    for i in range(len(tfids)):
        tfids[i] = list(tfids[i].values())
    return tfids

def update_uniqWords(uniqWords, token):
    for word in token:
        if word in uniqWords.keys():
            uniqWords[word] += 1
        else:
            uniqWords[word] = 1
    return uniqWords

def transform_message(message):
    message = message.lower()
    maketrans = str.maketrans
    translate_map = maketrans(constant.FILTER_STRING, constant.SPLIT * len(constant.FILTER_STRING))
    message = message.translate(translate_map)
    return message

def get_tfidf_data(messages):
    tokens = list()
    uniqWords = {}
    num_of_words = []
    for message in messages:
        message = transform_message(message)
        seq = message.split(constant.SPLIT)
        token = [stemmer.stem(word) for word in seq if word != '']
        tokens.append(token)
        uniqWords = update_uniqWords(uniqWords, token)

    num_of_words = get_num_of_words(tokens, uniqWords)
    tfidfs = get_tfidf(num_of_words, tokens, uniqWords)
    data = convert_tfids_to_data(tfidfs)
    return data

(messages, labels) = get_data()
data = get_tfidf_data(messages)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=0)
rc = RidgeClassifier(tol=1e-2, solver="sag")
rc.fit(data_train, labels_train)
labels_pred = rc.predict(data_test)
cr = classification_report(labels_test, labels_pred)
print(cr)