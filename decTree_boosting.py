import sys
import os
from globalConstants import *
import decTree
import copy

numBoostTrees = 5

#just does the boosting on the training space, ignoring any kind of privileged information..
def simpleBoost(trainData, numBoostTrees, datasetName):
    global nominalColumns
    
    gradThreshold = 0.1
    
    boostedTrees = []
    treeCount = 0
    trainData = copy.deepcopy(trainData) # make a deep copy of the original data, so that we are not changing the original data..
    stopBoosting = False
    while not stopBoosting:
        print "Getting a new Tree: ", len(boostedTrees)
        stopBoosting = True
        currTree = decTree.createTree(trainData, depth = 3, isClassifier = False, nominalColumns = nominalColumns[datasetName])
        #printTree(currTree)
        #compute the gradients with the new tree..
        newData = []
        for row in trainData:
            currValue = getRegressionValueOfTree(currTree, row)
            #compute the gradients..
            row[len(row)-1] = float(row[len(row)-1]) - currValue
            
            if row[len(row)-1] > gradThreshold:
                stopBoosting = False
            newData.append(row)

        #update the variable used for learning..
        trainData = newData
        treeCount += 1
        boostedTrees.append(currTree)
    print "Length of boostedTrees: ", len(boostedTrees)
    return boostedTrees

def offlineClusterBoost(trainData, clusters, numBoostTrees):
    boostedTrees = []

    return boostedTrees

def onlineClusterBoost(trainData, privTrainData, numBoostTrees):
    boostedTrees = []

    return boostedTrees

def getRegressionValueOfTree(tree, row):
    leaf = decTree.getRelevantLeafNode(tree, row)
    return numpy.mean(leaf.leafValues)
    
def getBoostResults(testData, boostedTrees, totalLabels):
    predictions = []
    for row in testData:
        res = 0
        for tree in boostedTrees:
           res += getRegressionValueOfTree(tree, row)
        
        closestLabel = None
        closestDistance = 1000000000000000
        for label in totalLabels:
            if abs(res - label) < closestDistance:
                closestDistance = abs(res - label)
                closestLabel = label

        tup = (closestLabel, row[len(row) - 1])
        predictions.append(tup)

    #return computeStats(predictions)
    return computeAccuracy(predictions), computeStats(predictions)

#The main function that calls all other functions, execution begins here
def main():
    global numBoostTrees
    global datasets
    global alpha
    global totalParts
    global splitCount
    global classLabels
    global nominalColumns

    for split in range(splitCount):
        '''
        print "\n"
        print "#"*40
        print "#"*40
        print "Printing results for split: ",(split+1)
        print "#"*40
        print "#"*40
        '''
        dirName = getSplitName(split)
        for datasetName in datasets:
             
            datasetName = "heart" #TODO: using this only for debugging..
            acc = 0
            print "Running "+datasetName+":"
            for part in range(totalParts):
                #"\nConstructing the privileged tree.."
                privTrainData = readData(datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv")
                privTree = None #TODO: remove this line..
                #privTree = decTree.createTree(privTrainData)
                
                #below are the train and test data in the pruned space..
                trainData = readData(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv")
                testData = readData(datasetName+"/"+dirName+"/pruned_test_"+str(part)+".csv")
                
                
                #get clusters for the training data using the privileged tree..
                '''
                clusters = {}
                for rowIndex in range(len(trainData)):
                    clusters[",".join(trainData[rowIndex])] = decTree.getClusterValue(privTrainData[rowIndex], privTree, nominalColumns[datasetName])
                '''
                boostedTrees = simpleBoost(trainData, numBoostTrees, datasetName)
                '''
                boostedTrees = offlineClusterBoost(trainData, clusters, numBoostTrees)
                boostedTrees = onlineClusterBoost(trainData, privTrainData, numBoostTrees)
                '''
                tup = getBoostResults(testData, boostedTrees, classLabels[datasetName])
                acc += tup[0]
                print "Accuracy is: ", tup[0]
                #print "Precision is: ", precision
                #print "Recall is: ", recall
                
                #exit() # TODO: exiting for debugging reasons..   
            print "Acg Accuracy is: ", acc/5.0
        exit()
#Execution begins here
if __name__ == "__main__" : main() 

