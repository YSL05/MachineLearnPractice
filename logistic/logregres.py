import numpy as np

def loaddataset():
    dataset = []
    labels  = []
    fr = open('TestSet.txt')

    for line in fr.readlines():
        linearr = line.strip().split()
        dataset.append([1.0, float(linearr[0]), float(linearr[1])])
        labels.append(int(linearr[2]))
    return dataset, labels

if __name__ == '__main__':
    dataset, labels = loaddataset()
    print(dataset)
    print(labels)
    