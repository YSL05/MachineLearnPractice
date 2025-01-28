# datatime: 2025/01/28
import numpy as np
from math import *
def loaddata():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1]
    return dataset, labels

def createvocablist(dataset):
    vocabset = set([])
    for data in dataset:
        vocabset = vocabset | set(data)
    return list(vocabset)

def setofworld2vec(vocabset, data):
    vec = [0] * len(vocabset)
    for word in data:
        if word in vocabset:
            vec[vocabset.index(word)] = 1
        else:
            print ("the world: %s is not in my vocabularry!" % word)
    return vec

def trainNB0(dataset, labels):
    datasetsize = len(labels)
    vocabsize = len(dataset[0])
    pAbusive = sum(labels) / float(datasetsize)
    p0num = np.ones(vocabsize)
    p1num = np.ones(vocabsize)
    p0denom = 2.0
    p1denom = 2.0
    for i in range(datasetsize):
        if labels[i] == 1:
            p1num = p1num + dataset[i]
            p1denom = p1denom + sum(dataset[i])
        else:
            p0num = p0num + dataset[i]
            p0denom = p0denom + sum(dataset[i])
    p0vect = p0num / p0denom
    p1vect = p1num / p1denom
    return p0vect, p1vect, pAbusive

if __name__ == "__main__":
    dataset, labels = loaddata()
    vocabset = createvocablist(dataset)
    datamatrix = []
    for data in dataset:
        datamatrix.append(setofworld2vec(vocabset, data))
    p0v, p1v, pab = trainNB0(datamatrix, labels)
    print(vocabset)
    print(p0v)
    print(p1v)
    print(pab)
    
