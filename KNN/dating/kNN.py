import numpy as np
import operator
import matplotlib.pyplot as plt

def file2matrix(filename):
    '''
    transfer the file to dataset and label
    '''
    fr = open(filename)
    filelines = fr.readlines()
    numberoflines = len(filelines)
    returnmat = np.zeros((numberoflines, 3))
    classlabelvector = []
    index = 0
    for line in filelines:
        line = line.strip()
        listfromline = line.split('\t')
        for i in range(3):
            returnmat[index, i] = float(listfromline[i])
        if listfromline[-1] == 'largeDoses':
            classlabelvector.append(3)
        if listfromline[-1] == 'didntLike':
            classlabelvector.append(1)
        if listfromline[-1] == 'smallDoses':
            classlabelvector.append(2)
        index = index + 1
    return returnmat, classlabelvector

def showdataset(dataset, labels):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    labelcolor = []
    for i in labels:
        if i == 1:
            labelcolor.append('black')
        if i == 2:
            labelcolor.append('orange')
        if i == 3:
            labelcolor.append('red')
    # print scatter figure
    axs[0][0].scatter(dataset[:,0], dataset[:,1], color=labelcolor, s=15, alpha=.5)
    axs[0][0].set_title('fly vs game')
    axs[0][0].set_xlabel('fly')
    axs[0][0].set_ylabel('game')

    axs[0][1].scatter(dataset[:,0], dataset[:,2], color=labelcolor, s=15, alpha=.5)
    axs[0][1].set_title('fly vs ice cream')
    axs[0][1].set_xlabel('fly')
    axs[0][1].set_ylabel('ice cream')

    axs[1][0].scatter(dataset[:,1], dataset[:,2], color=labelcolor, s=15, alpha=.5)
    axs[1][0].set_title('game vs ice cream')
    axs[1][0].set_xlabel('game')
    axs[1][0].set_ylabel('ice cream')

    plt.show()

def autonorm(dataset):
    '''
    norm the maritx
    '''
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    normdataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minvals, (m, 1))
    normdataset = normdataset / np.tile(ranges, (m, 1))
    return normdataset, ranges, minvals

def classify(inx, dataset, labels, k):
    
    datasetsize = dataset.shape[0]
    diffmat = np.tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdistace = sqdiffmat.sum(axis=1)
    distance = sqdistace ** 0.5

    sorteddistanceindex = distance.argsort()
    classlabels = {}
    for i in range(k):
        votelabel = labels[sorteddistanceindex[i]]
        classlabels[votelabel] = classlabels.get(votelabel, 0) + 1
    sortedclasslabels = sorted(classlabels.items(), key = operator.itemgetter(1), reverse=True)
    return sortedclasslabels[0][0]

def datingclasstest():
    
    filename = 'datingTestSet.txt'
    dataset, labels = file2matrix(filename)
    
    horatio = 0.10
    normdataset, ranges, minvals = autonorm(dataset)

    m = dataset.shape[0]
    numtestvecs = int(m*horatio)

    errorcount = 0

    for i in range(numtestvecs):
        classlabelresult = classify(normdataset[i,:], normdataset[numtestvecs:m,:],
                                    labels[numtestvecs:m], 4)
        print('分类结果：%d\t真实类别 %d' % (classlabelresult, labels[i]))
        if classlabelresult != labels[i]:
            errorcount = errorcount + 1
    print('错误了：%f%%' % (errorcount/float(numtestvecs)*100))

def classperson():
    result = ['didntLike', 'smallDoses', 'largeDoses']
    precenttats = float(input("time of playing game: "))
    ffmiles = float(input("frequent flier miles earned per year: "))
    icecream = float(input("liters of icecream consumed per year: "))
    filename = 'datingTestSet.txt'
    dataset, labels = file2matrix(filename)
    normdataset, ranges, minvals = autonorm(dataset)

    inarr = [ffmiles, precenttats, icecream]
    classifierresult = classify((inarr - minvals) / ranges, normdataset, labels, 3)
    print("you will probably like this person: ", result[classifierresult - 1])
        
if __name__ == '__main__':
    # open file and get dataset with labels
    # file = 'datingTestSet.txt'
    # dataset, labels = file2matrix(file)

    # show the scatter of dataset
    # showdataset(dataset, labels)
    
    # norm dataset
    # normdataset, ranges, minvals = autonorm(dataset)
    # print(normdataset)
    # print(ranges)
    # print(minvals)

    # datingclasstest()

    classperson()



