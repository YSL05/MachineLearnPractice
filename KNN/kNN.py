'''
sample classify
'''
import numpy as np
import operator
import matplotlib.pyplot as plt

def createdataset():
    dataset = np.array([[1.0, 1.0], [1.1, 1.0], [0.0, 0.0], [0.1, 0.0]])
    labels = ['A', 'A', 'B', 'B']
    return dataset, labels
def showdataset(dataset, labels):
    plt.scatter(np.transpose(dataset)[0], np.transpose(dataset)[1])
    plt.show()
    
def classify(inx, dataset, labels, k):
    # calcute the distance between inx with dataset
    datasetsize = dataset.shape[0]
    diffmat = np.tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdistances = sqdiffmat.sum(axis = 1)
    distances = sqdistances ** 0.5
    # find kth smallest datapoint's label  
    sortdistaceindex = distances.argsort()
    classlabel = {}
    for i in range(k):
        votelabel = labels[sortdistaceindex[i]]
        classlabel[votelabel] = classlabel.get(votelabel, 0) + 1
    # return the most hit label in kth datapoint
    sortclasslabel = sorted(classlabel.items(), key = operator.itemgetter(1), reverse=True)
    return sortclasslabel[0][0]

if __name__ == '__main__':
    # creat dataset
    group, labels = createdataset()
    showdataset(group, labels)
    # creat testcase
    test = [1.2, 1]
    # classify
    testlabel = classify(test, group, labels, 3)
    print(testlabel)


