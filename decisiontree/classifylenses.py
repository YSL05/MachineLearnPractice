import trees
import decisionTreePlot

fr = open('lenses.txt')

lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenseslabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensestree = trees.createtree(lenses, lenseslabels)

print(lensestree)
decisionTreePlot.createPlot(lensestree)
