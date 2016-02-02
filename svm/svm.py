import sys
import os
currPath = os.getcwd()
baseName = os.path.basename(currPath)
sys.path.insert(0, currPath[:-len(baseName)])

from globalConstants import *
from algorithms import *


def convertClassLabel(cl):
    if cl == 0:
        return float(-1)
    else:
        return float(1)

accuracy = {}
for dataset in datasets:
    accuracy[dataset] = []    
    print "*"*40
    print "Running Experiments on dataset: ",dataset
    print "*"*40
    for split_num in range(splitCount):
        partAccuracy = 0
        for part in range(totalParts):
            trainFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_train_"+str(part)+".csv", "r").readlines()
            testFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_test_"+str(part)+".csv", "r").readlines()
                        
            X = []
            Y = []
            Xtest = []
            Ytest = []
            for row in trainFile:
                row = row.split(",")
                row = [ float(r) for r in row]
                X.append(row[:-1])
                
                #changing the class labels to -1 and 1
                Y.append(convertClassLabel(row[len(row) - 1]))

            
            for row in testFile:
                row = row.split(",")
                row = [ float(r) for r in row]
                Xtest.append(row[:-1])
                Ytest.append(convertClassLabel(row[len(row) - 1]))
            
            svm_algo = SVM()
            svm_algo.learn(X, Y)
            predictions = svm_algo.predict(Xtest)                            
            print predictions
            numCorrectPredictions = 0
            for rowIndex in range(len(Ytest)):
                if predictions[rowIndex]*Ytest[rowIndex] > 0:
                    numCorrectPredictions+=1

            print "Accuracy is: ", (numCorrectPredictions*1.0)/len(Ytest)
            partAccuracy += (numCorrectPredictions*1.0)/len(Ytest)

            '''
            print "*"*40
            print "Printing Predicted\n"
            print "*"*40
            print Xtest.shape
            print predicted.shape
            '''
        accuracy[dataset].append(partAccuracy/totalParts)
    
    print "Accuracy for dataset: ", dataset,": ", round(numpy.mean(accuracy[dataset]), 4), "+- ", round(numpy.std(accuracy[dataset]), 4)
  
