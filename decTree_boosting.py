import sys
import os
from globalConstants import *
import decTree

numBoostTrees = 5



def offlineClusterBoost(trainData, clusters, numBoostTrees):
    boostedTrees = []

    return boostedTrees

def onlineClusterBoost(trainData, privTrainData, numBoostTrees):
    boostedTrees = []

    return boostedTrees

def getBoostResults(testData, boostedTrees):
    boostedTrees = []
    
    res = 0
    
    for row in testData:
        for tree in boostedTrees:
           res = getValueTree() 
    return boostedTrees


#The main function that calls all other functions, execution begins here
def main():
    global numBoostTrees
    global datasets
    global alpha
    global totalParts
    global splitCount
    
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
            #print ""
            #print "#"*40
            print "Running "+datasetName+":"
            for part in range(totalParts):
                #"\nConstructing the privileged tree.."
                privTrainData = readData(datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv")
                privTree = createTree(privTrainData, 15)
                
                #below are the train and test data in the pruned space..
                trainData = readData(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv")
                testData = readData(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv")
                
                #get clusters for the training data using the privileged tree..
                clusters = {}        
                for rowIndex in range(len(trainData)):
                    clusters[",".join(trainData[rowIndex])] = decTree.getClusterValue(privTrainData[rowIndex], privTree)
                
                boostedTrees = offlineClusterBoost(trainData, clusters, numBoostTrees)
                boostedTrees = onlineClusterBoost(trainData, privTrainData, numBoostTrees)
                getBoostResults(testData, boostedTrees)
                
                exit() # TODO: exiting for debugging reasons..                
#Execution begins here
if __name__ == "__main__" : main() 

