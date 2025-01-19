import numpy as np
import operator
from os import listdir
def img2vector(filename):
    returnvect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnvect[0, 32 * i + j] = int(line[j])
    return returnvect

def createdataset():
    trainingfileslist = listdir('trainingdigits')
    m = len(trainingfileslist)
    dataset = np.zeros((m, 1024))
    labels = []
    for i in range(m):
        filename = trainingfileslist[i]
        classlabel = int(filename.split('_')[0])
        labels.append(classlabel)
        dataset[i,:] = img2vector('trainingdigits/%s' %(filename))
    return dataset, labels

def classify(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = np.tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances ** 0.5

    sortdistanceindex = distances.argsort()
    classlabels = {}
    for i in range(k):
        votelabel = labels[sortdistanceindex[i]]
        classlabels[votelabel] = classlabels.get(votelabel, 0) + 1
    sortedclasslables = sorted(classlabels.items(), key = operator.itemgetter(1), reverse = True)
    return sortedclasslables[0][0]
def testclassify():
    testdigitsfileslist = listdir('testdigits')
    m = len(testdigitsfileslist)
    errorcount = 0
    dataset, labels = createdataset()

    for i in range(m):
        filename = testdigitsfileslist[i]
        acturelabel = int(filename.split('_')[0])
        inx = img2vector('testdigits/%s' %(filename))
        classresult = classify(inx, dataset, labels, 3)
        print("真实结果是：%s\t预测结果是：%s" %(acturelabel, classresult))
        if (classresult != acturelabel):
            errorcount = errorcount + 1
    print("总共错误了%d个数据\n错误率为: %f%%" %(errorcount, errorcount/m * 100))

if __name__ == '__main__':
    testclassify()
