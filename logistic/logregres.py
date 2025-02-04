import numpy as np
from math import exp
def loaddataset():
    dataset = []
    labels  = []
    fr = open('TestSet.txt')

    for line in fr.readlines():
        linearr = line.strip().split()
        dataset.append([1.0, float(linearr[0]), float(linearr[1])])
        labels.append(int(linearr[2]))
    return dataset, labels

def sigmod(inx):
    return 1.0 / (1 + np.exp(-inx))

def gradascent(dataset, labels):
    datasetmatrix = np.asmatrix(dataset)
    labelsmatrix  = np.asmatrix(labels).transpose()
    m,n = np.shape(datasetmatrix)
    alpha = 0.001
    maxcycle = 500
    weights = np.ones((n,1))

    for k in range(maxcycle):
        h = sigmod(datasetmatrix * weights)
        error = (labelsmatrix - h)
        weights = weights + alpha * datasetmatrix.transpose() * error
    return weights


if __name__ == '__main__':
    dataset, labels = loaddataset()
    weights = gradascent(dataset, labels)
    print(weights)