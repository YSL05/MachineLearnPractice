# datatime: 2025/01/28
import numpy as np
from math import *
def loaddata():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
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
    for i in range(datasetsize):
        if labels[i] == 1:
            p1num = p1num + dataset[i]
        else:
            p0num = p0num + dataset[i]
    p0vect = [log(i / sum(p0num)) for i in p0num]
    p1vect = [log(i / sum(p0num)) for i in p1num]
    return p0vect, p1vect, pAbusive

def classifyNB(vect, p0vect, p1vect, pAbusive):
    p1 = sum(vect * p1vect) + log(pAbusive)
    p0 = sum(vect * p0vect) + log(1.0 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    # get training dataset
    dataset, labels = loaddata()
    vocabset = createvocablist(dataset)
    datamatrix = []
    for data in dataset:
        datamatrix.append(setofworld2vec(vocabset, data))
    # training the modle
    p0v, p1v, pab = trainNB0(datamatrix, labels)

    # test the modle
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setofworld2vec(vocabset, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pab))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setofworld2vec(vocabset, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pab))     

if __name__ == "__main__":
    testingNB()
    
