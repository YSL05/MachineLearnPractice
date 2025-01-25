from math import log

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
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'yes'], [0, 1, 'no'], [0, 1, 'no']]
    return dataset

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

if __name__ == '__main__':
    dataset = creatdataset()
    shannonent = calshannonent(dataset)
    print(shannonent)
    print(choosebestfeaturetosplit(dataset))

