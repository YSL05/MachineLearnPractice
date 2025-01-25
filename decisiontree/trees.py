from math import log
import operator

def calshannonent(dataset):
    datasetsize = len(dataset)
    labelcount = {}
    for data in dataset:
        label = data[-1]
        labelcount[label] = labelcount.get(label, 0) + 1
    shannonent = 0
    for key in labelcount:
        prob = float(labelcount[key])/datasetsize
        shannonent = shannonent - prob * log(prob,2)
    return shannonent

def creatdataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    feature = ['no surfacing', 'flippers']
    return dataset, feature

def splitdataset(dataset, axis, value):
    retdataset = []
    for data in dataset:
        if data[axis] == value:
            reducedata = data[ : axis]
            reducedata.extend(data[axis + 1 : ])
            retdataset.append(reducedata)
    return retdataset

def choosebestfeaturetosplit(dataset):
    numfeature = len(dataset[0]) - 1
    baseentropy = calshannonent(dataset)
    bestinfogain = 0
    bestfeature = -1
    for i in range(numfeature):
        featlist = [example[i] for example in dataset]
        uniqueval = set(featlist)
        newebtropy = 0
        for value in uniqueval:
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset) / float(len(dataset))
            newebtropy = newebtropy + prob * calshannonent(subdataset)
        infogain = baseentropy - newebtropy
        if (infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature = i
    return bestfeature

def majoritycnt(labels):
    labelcount = {}
    for vote in labels:
        labelcount[vote] = labelcount.get(vote, 0) + 1
    sortedlabelcount = sorted(labelcount, key=operator.itemgetter(1), reverse=True)
    return sortedlabelcount[0][0]

def createtree(dataset, feature):
    labellist = [example[-1] for example in dataset]
    if labellist.count(labellist[0]) == len(labellist):
        return labellist[0]
    if len(dataset[0]) == 1:
        return majoritycnt(labellist)
    bestfeature = choosebestfeaturetosplit(dataset)
    bestfeaturelabel = feature[bestfeature]
    mytree = {bestfeaturelabel:{}}
    del(feature[bestfeature])
    featurevalue = [example[bestfeature] for example in dataset]
    uniqueval = set(featurevalue)
    for value in uniqueval:
        sublabels = feature[:]
        mytree[bestfeaturelabel][value] = createtree(splitdataset(dataset, bestfeature, value), sublabels)
    return mytree

def classify(inputtree, featurelabel, testvec):
    firststr = list(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featurelabel.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__ == 'dict':
                classlabel = classify(seconddict[key], featurelabel, testvec)
            else:
                classlabel = seconddict[key]
    return classlabel
def storeTree(inputtree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputtree, fw)
    fw.close()

def grabtree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataset, feature = creatdataset()
    mytree = createtree(dataset, feature)
    storeTree(mytree, 'decisiontree.txt')
    mytree = 0
    mytree = grabtree('decisiontree.txt')
    print(mytree)
    dataset, feature = creatdataset()
    print(classify(mytree, feature, [1, 1]))


