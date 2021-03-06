import sys
import os
from globalConstants import *
import decTree
import copy

def computeOfflineGradient(tree, row, nominalColumns):
    currValue = getLeafRegressionValueOfTree(tree, row, nominalColumns)
    #compute the gradients..
    return float(row[len(row)-1]) - stepSize*currValue
    
stepSize = 0.1
iterativeStepSize = 0.4

def boosting(trainData, datasetName, privTrainData, isSimple, isOffline, alpha = 0, isOnline = False):
    global maxBoostingTrees
    global stepSize
    prevError = 10000000000
    currError = None
    global prunedNominalColumns    
    global privNominalColumns
    gradThreshold = 0.1
    origData = trainData
    origPrivData = privTrainData
    boostedTrees = []
    treeCount = 0

    trainData = copy.deepcopy(trainData) # make a deep copy of the original data, so that we are not changing the original data..
    if privTrainData != None:
        privTrainData = copy.deepcopy(privTrainData) # make a deep copy of the original data, so that we are not changing the original data..
    stopBoosting = False
    classIndex = len(origData[0]) - 1
    predictions = [ 0 for p in range(len(origData))]

    privTree = None
    clusters = None        
    
    if isOffline:
        privTree = decTree.createTree(privTrainData, nominalColumns = privNominalColumns[datasetName])
        clusters = getPrivTreeClusters(privTrainData, privTree, trainData, datasetName)

    while (not stopBoosting and len(boostedTrees) < maxBoostingTrees):
        currPrivTree = None
        currTree = None
        #print "Getting a new Tree: ", len(boostedTrees)
        stopBoosting = True
        if isSimple:
            currTree = decTree.createTree(trainData, depth = 4, isClassifier = False, nominalColumns = prunedNominalColumns[datasetName])
        elif isOffline:
            currTree = decTree.createTree(trainData, depth = 4, isClassifier = False, isPrivAvailable = True, cluster = clusters, nominalColumns = prunedNominalColumns[datasetName], alpha = alpha)
        elif isOnline:
            currTree = decTree.createTree(trainData, depth = 4, isClassifier = False, isPrivAvailable = False, cluster = clusters, nominalColumns = prunedNominalColumns[datasetName], alpha = alpha)      
            currPrivTree = decTree.createTree(privTrainData, depth = 4, isClassifier = False, isPrivAvailable = False, cluster = clusters, nominalColumns = privNominalColumns[datasetName], alpha = alpha)

        newData = []
        #compute the gradients with the new tree..
        newPrivData = []
        
        count = 0
        for i in range(len(trainData)): 
            row = trainData[i]
            currValue = getLeafRegressionValueOfTree(currTree, row, prunedNominalColumns[datasetName])
            if isOnline:
                p_priv = getLeafRegressionValueOfTree(currPrivTree, privTrainData[i], privNominalColumns[datasetName])
                
                #trainData[i][len(trainData[i]) - 1] = float(trainData[i][len(trainData[i])-1]) - stepSize*(currValue + iterativeStepSize*(p_priv - currValue))
                #privTrainData[i][len(privTrainData[i]) - 1] = float(privTrainData[i][len(privTrainData[i]) - 1]) - stepSize*(p_priv + iterativeStepSize*(currValue - p_priv))

                trainData[i][len(trainData[i]) - 1] = float(trainData[i][len(trainData[i])-1]) - stepSize*(currValue + alpha*(p_priv - currValue))
                privTrainData[i][len(privTrainData[i]) - 1] = float(privTrainData[i][len(privTrainData[i]) - 1]) - stepSize*(p_priv + alpha*(currValue - p_priv))

                #make of a new copy for the new runs..
                newData.append(trainData[i])
                newPrivData.append(privTrainData[i])
                
            else:          
                row[len(row)-1] = computeOfflineGradient(currTree, row, prunedNominalColumns[datasetName])
                newData.append(row)
            predictions[i] += stepSize*currValue
            #predictions[i] += stepSize*row[len(row)-1]

        #update the variable used for learning..
        trainData = newData
        privTrainData = newPrivData

        #compute overall error..
        currError = 0
        for index in range(len(origData)):
            #compute the MSE error..
            currError = currError + ((float(origData[index][classIndex]) - float(predictions[index]))*(float(origData[index][classIndex]) - float(predictions[index])))
        
        if abs(prevError - currError) > 0.01*len(origData):
            stopBoosting = False
        prevError = currError
        treeCount += 1
        boostedTrees.append(currTree)
    print "Length of boostedTrees: ", len(boostedTrees)
    return boostedTrees

#just does the boosting on the training space, ignoring any kind of privileged information..
def simpleBoost(trainData, datasetName):
    return boosting(trainData = trainData, datasetName = datasetName, isSimple = True, isOffline = False, privTrainData = None, alpha = 0)

def offlineClusterBoost(trainData, privTrainData, datasetName, alpha):
    return boosting(trainData = trainData, datasetName = datasetName, isSimple = False, isOffline = True, privTrainData = privTrainData, alpha = alpha)

def onlineClusterBoost(trainData, privTrainData, datasetName, alpha):
    return boosting(trainData = trainData, datasetName = datasetName, isSimple = False, isOffline = False, isOnline = True, privTrainData = privTrainData, alpha = alpha)

def getLeafRegressionValueOfTree(tree, row, nominalColumns):
    leaf = decTree.getRelevantLeafNode(tree, row, nominalColumns)
    return numpy.mean(leaf.leafValues)
    
def getPrivTreeClusters(privTrainData, privTree, trainData, datasetName):           
    #get clusters for the training data using the privileged tree..                
     #"\nConstructing the privileged tree.."
    global privNominalColumns
    global prunedNominalColumns
    
    decTree.assignClustersToLeaves(privTree)

    clusters = {}
    diffClusters = []
    for rowIndex in range(len(trainData)):
        prunedRow = trainData[rowIndex]
        prunedRow = prunedRow[:-1] #remove the class label from the row..
        clusters[",".join(prunedRow)] = decTree.getClusterValue(privTrainData[rowIndex], privTree, privNominalColumns[datasetName])


    for key in clusters:
        if clusters[key] not in diffClusters:
            diffClusters.append(clusters[key])
    print "Total Number of clusters: ", len(diffClusters)
    return clusters

def getBoostResults(testData, boostedTrees, totalLabels, nominalColumns):
    global stepSize
    predictions = []
    for row in testData:
        res = 0
        for tree in boostedTrees:
           res += stepSize*getLeafRegressionValueOfTree(tree, row, nominalColumns)
        
        closestLabel = None
        closestDistance = 1000000000000000
        for label in totalLabels:
            if abs(res - label) < closestDistance:
                closestDistance = abs(res - label)
                closestLabel = label

        tup = (closestLabel, row[len(row) - 1])
        predictions.append(tup)

    res = computeStats(predictions)
    #return computeStats(predictions)
    return (computeAccuracy(predictions), res[0], res[1], res[2]) 

#The main function that calls all other functions, execution begins here
def main():
    global datasets
    global alpha
    global totalParts
    global splitCount
    global classLabels
    global nominalColumns
    global privNominalColumns
    global prunedNominalColumns
    
    splitOldAccuracy = {}
    splitOldPrecision = {}
    splitOldRecall = {}
    
    splitAccuracy = {}
    splitPrecision = {}
    splitRecall = {}

    for datasetName in datasets:
        splitAccuracy[datasetName] = []
        splitPrecision[datasetName] = {}
        splitRecall[datasetName] = {}
        
        splitOldAccuracy[datasetName] = []
        splitOldPrecision[datasetName] = {}
        splitOldRecall[datasetName] = {}

    init() #inits some global variables required for the execution..
    #print prunedNominalColumns
    #print privNominalColumns 
    print privNominalColumns
    print prunedNominalColumns
    
    for split in range(splitCount):
        print "\n"
        print "#"*40
        print "#"*40
        print "Printing results for split: ",split
        print "#"*40
        print "#"*40
        dirName = getSplitName(split)
        for datasetName in datasets:
            print ""
            print "#"*40
            print "Running "+datasetName+":"
            normalAcc = 0
            privAcc = 0
            newAcc = []
            normalPrecision = {}
            normalRecall = {}
            normalAccuracy = {}

            newPrecision = []
            newRecall = []
            #normalAccuracy = {}
            
            for part in range(totalParts):
                print "Running "+datasetName+" with fold: ", part
                
                print "\nRunning pruned dataset"
 
                #below are the train and test data in the pruned space..
                trainData = readData(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv")
                testData = readData(datasetName+"/"+dirName+"/pruned_test_"+str(part)+".csv")

                boostedTrees = simpleBoost(trainData, datasetName)
                currNormalAcc, precision, recall, accuracy = getBoostResults(testData, boostedTrees, classLabels[datasetName], prunedNominalColumns[datasetName])
                print currNormalAcc
                normalAcc += currNormalAcc
                for label in precision:
                    if label not in normalPrecision:
                        normalPrecision[label] = 0
                    normalPrecision[label] += precision[label]
                    
                    if label not in normalRecall:
                        normalRecall[label] = 0
                    normalRecall[label] += recall[label]
                    
                    if label not in normalAccuracy:
                        normalAccuracy[label] = 0
                    normalAccuracy[label] += accuracy[label]

                isOffline = False
                #read the priv training data..
                privTrainData = readData(datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv")
                
                newAcc.append([])
                newPrecision.append([])
                newRecall.append([])
                for run in range(20):
                    alpha = (run+1)/10.0
                    print "Running the new logic with alpha = ",alpha
                    if isOffline:
                        boostedTrees = offlineClusterBoost(trainData, privTrainData, datasetName, alpha)
                    else:
                        boostedTrees = onlineClusterBoost(trainData, privTrainData, datasetName, alpha)

                    currAcc, precision, recall, accuracy = getBoostResults(testData, boostedTrees, classLabels[datasetName], prunedNominalColumns[datasetName])   
                    print currAcc,"\t",alpha
                    '''
                    if not isOffline:
                        for i in range(20):
                            newAcc[part].append(currAcc)
                            newPrecision[part].append(precision)
                            newRecall[part].append(recall)
                        break    
                    '''
                    #break
                    newAcc[part].append(currAcc)
                    newPrecision[part].append(precision)
                    newRecall[part].append(recall)
                #print "#"*40
                #print ""
            print "\nNormal Accuracy is", normalAcc/float(totalParts)
            splitOldAccuracy[datasetName].append(normalAcc/float(totalParts))
            
            #print "Privileged Accuracy is", privAcc/float(totalParts)
            for label in normalPrecision:
                print "Stats for label: ",label
                print "\tPrecision is: ", normalPrecision[label]/float(totalParts)
                print "\tRecall is: ", normalRecall[label]/float(totalParts)

                print "-"*30
                
                if label not in splitPrecision[datasetName]:
                    splitOldPrecision[datasetName][label] = []
                if label not in splitRecall[datasetName]:
                    splitOldRecall[datasetName][label] = []
                
                splitOldPrecision[datasetName][label].append(normalPrecision[label]/float(totalParts))
                splitOldRecall[datasetName][label].append(normalRecall[label]/float(totalParts))
                
            avgAcc = [0 for i in range(20)]
            avgPrecision = [ {} for i in range(20)]
            avgRecall = [ {} for i in range(20) ]
            for run in range(20):
                for j in range(totalParts):
                    avgAcc[run] += newAcc[j][run]
                    for lbl in newPrecision[j][run]:
                        if lbl not in avgPrecision[run]:
                             avgPrecision[run][lbl] = 0
                        avgPrecision[run][lbl] += newPrecision[j][run][lbl]
                        
                        if lbl not in avgRecall[run]:
                             avgRecall[run][lbl] = 0
                        avgRecall[run][lbl] += newRecall[j][run][lbl]     

            maxAvgAccuracy = 0
            maxAlpha = 0
            chosenI = 0
            for i in range(20):
                #print "Accuracy for run - ",i,": ", avgAcc[i] 
                if  maxAvgAccuracy < avgAcc[i]:
                    chosenI = i
                    maxAlpha = float(i+1)/10
                    maxAvgAccuracy = avgAcc[i]

            print "\nNew Accuracy is: ",maxAvgAccuracy/float(totalParts), "for alpha: ",maxAlpha
            splitAccuracy[datasetName].append(maxAvgAccuracy/float(totalParts))
            
            for lbl in normalAccuracy:
                print "Stats for label: ",lbl
                print "\tPrecision for the chosen alpha is: ", avgPrecision[chosenI][lbl]/float(totalParts)
                print "\tRecall for the chosen alpha is: ", avgRecall[chosenI][lbl]/float(totalParts)
                print "-"*30
                if lbl not in splitPrecision[datasetName]:
                    splitPrecision[datasetName][lbl] = []
                if lbl not in splitRecall[datasetName]:
                    splitRecall[datasetName][lbl] = []
                splitPrecision[datasetName][lbl].append(avgPrecision[chosenI][lbl]/float(totalParts))
                splitRecall[datasetName][lbl].append(avgRecall[chosenI][lbl]/float(totalParts))
               
    print "-"*40
    print "-"*40
    print "-"*40            
    for datasetName in datasets:
        print "Printing Results for Dataset: ", datasetName
        print "Avg Old Accuracy: ", round(numpy.mean(splitOldAccuracy[datasetName]), 4), "+- ", round(numpy.std(splitOldAccuracy[datasetName]), 4)
        print "Avg. New Accuracy: ", round(numpy.mean(splitAccuracy[datasetName]), 4), "+- ", round(numpy.std(splitAccuracy[datasetName]), 4)
        
        for lbl in splitPrecision[datasetName]:
            print "Stats for label: ", lbl
            print "\t Old Avg. Precision is: ", round(numpy.mean(splitOldPrecision[datasetName][lbl]), 4), "+- ", round(numpy.std(splitOldPrecision[datasetName][lbl]), 4)
            print "\t New Avg. Precision is: ", round(numpy.mean(splitPrecision[datasetName][lbl]), 4), "+- ", round(numpy.std(splitPrecision[datasetName][lbl]), 4)
            print
            print "\t Old Avg. Recall is: ", round(numpy.mean(splitOldRecall[datasetName][lbl]), 4), "+- ", round(numpy.std(splitOldRecall[datasetName][lbl]), 4)
            print "\t New Avg. Recall is: ", round(numpy.mean(splitRecall[datasetName][lbl]), 4), "+- ", round(numpy.std(splitRecall[datasetName][lbl]), 4)
            print
        
#Execution begins here
if __name__ == "__main__" : main() 

