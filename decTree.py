#!/usr/bin/python
import sys
import csv
from math import *
from Node import Node
from computeStats import *
from globalConstants import *#file containing all the global constants..


def calcPrivInfoGain(currentEntropy, clusterEntropy, subDataSet1,subDataSet2, isClassifier, cluster):
    global numClusters
    global alpha

    normalGain = calcInfoGain(currentEntropy, subDataSet1, subDataSet2, isClassifier)
    p = float(len(subDataSet1))/(len(subDataSet1)+len(subDataSet2))
    privGain = clusterEntropy -p*calcPrivEntropy(subDataSet1, cluster) - (1-p)*calcPrivEntropy(subDataSet2, cluster)
    
    return (normalGain, privGain)

def calcPrivEntropy(data, cluster):
    classLabelCounts = {}
    for row in data:
        c = cluster[",".join(row)]
        if c in classLabelCounts:
            classLabelCounts[c] += 1
        else:
            classLabelCounts[c] = 1

    totalRows = len(data)
    entropy = 0.0

    for key in classLabelCounts:
        p = float(classLabelCounts[key])/totalRows
        entropy -= p*log(p,2)
    return entropy

'''
The createTree function is where all the magic happens, 
We call createTree recursively until we reach the required depth or a good decision tree
The method takes a sub part of the dataset as input and creates a tree based on the decision criteria.

'''
def createTree(subDataSet, depth=15,threshold=0.0, isPrivAvailable = False, isClassifier = True, cluster = {}, nominalColumns = []):
    global limitGainBounds
    global allowShift
    #print nominalColumns
    #Counting the number of rows in the Dataset
    numOfRows = len(subDataSet)

    #if the required depth is > 0 and the dataset has some rows 
    if depth > 0 and len(subDataSet) > 0:

        #We first calculate the entropy for the entire data set
        if isClassifier:
            entropy = calcEntropy(subDataSet)
        else:
            entropy = calcVariance(subDataSet)
        
        #We initially set the best parameters to 0 and None
        bestGain = threshold
        bestSet = None
        bestCriteria = None
        bestColumn = None

        #Lets first count the number of columns, excluding the last column (Ofcourse :p )
        numberOfColumns = len(subDataSet[0])-1

        #Now we iterate through each column to see which is the best column to split on
        for col in range(0,numberOfColumns):

            #We then see which values are present in the column, we will choose one vlalue as criteria to split into 2 datasets
            valuesInColumn = {}
            for row in subDataSet:
                valuesInColumn[row[col]]=1  

            privGainList = []
            normalAvg = 0
            privAvg = 0
            
            normalMax = -1000
            normalMin = 100000
            
            privMax = -1000
            privMin = 100000
            
            #We are now iterating through each value in the current iteration of column to see which value serves as the best split
            for value in valuesInColumn:
                #Split the dataset on the current value of column and value
                (set1,set2) = splitData(subDataSet,col, value, nominalColumns)
                if len(set1) > 0 and len(set2) > 0:
                    #Calculate infoGain for each col and each value in the column
                    if isPrivAvailable == False:
                        infoGain = calcInfoGain(entropy, set1,set2, isClassifier)
                        #Choose the best col and value 
                        if infoGain > bestGain and len(set1) > 0 and len(set2) > 0 :
                            bestGain = infoGain
                            bestSet = (set1, set2)
                            bestCriteria = value
                            bestColumn = col
                    else:
                        #TODO: check priv entropy vs. variance..
                        currGain, privGain = calcPrivInfoGain(entropy, calcPrivEntropy(subDataSet, cluster), set1,set2, isClassifier, cluster)
                        privGainList.append((currGain, privGain, (set1, set2), value, col))
                        
                        if currGain >= threshold: #only decent gains are included in the calculation..
                            normalAvg += currGain
                            
                            
                            #update the min and max of the training Gains..
                            if currGain > normalMax:
                                normalMax = currGain
                            if currGain < normalMin:
                                normalMin = currGain
                                
                            #update the min and max of the priv Gains...
                            if privGain > privMax:
                                privMax = privGain
                            if privGain < privMin:
                                privMin = privGain
                             
                            privAvg += privGain
                
            if isPrivAvailable == True:
                #'''
                if len(privGainList) == 0:
                    continue # if all the rows have the same value for this column... (very very rare..)
                normalAvg = normalAvg/len(privGainList)
                privAvg = privAvg/len(privGainList)

                shift = abs(normalAvg - privAvg)
                    
                if privAvg > normalAvg:
                    index = 0
                else:
                    index = 1
                normalRange = normalMax - normalMin
                privRange = privMax - privMin
                #print normalRange, privRange
                for ind in range(len(privGainList)):
                    currTuple = privGainList[ind]

                    #limit the bounds of the gains..
                    if limitGainBounds:
                        if normalMin == normalMax:
                            currTuple = (0.5, currTuple[1], currTuple[2], currTuple[3], currTuple[4])
                        else:
                            currTuple = ( float( currTuple[0] - normalMin)/normalRange, currTuple[1], currTuple[2], currTuple[3], currTuple[4])
                        if privMin == privMax:
                            currTuple = (currTuple[0], 0.5, currTuple[2], currTuple[3], currTuple[4])
                        else:
                            currTuple = ( currTuple[0], float( currTuple[1] - privMin)/privRange, currTuple[2], currTuple[3], currTuple[4])
                    else:                                            
                        if index == 0:
                            currTuple = (currTuple[0]+ abs(shift), currTuple[1], currTuple[2], currTuple[3], currTuple[4])
                        else:
                            currTuple = (currTuple[0], currTuple[1]+ abs(shift), currTuple[2], currTuple[3], currTuple[4])
                
                #'''
                #find the max threshold
                for tup in privGainList:
                    currGain = combineGain(tup[0], tup[1], isClassifier)
                    if currGain > bestGain and len(tup[2][0]) > 0 and len(tup[2][1]) > 0:
                        #print tup[0],"\t",tup[1],"\t", currGain
                        bestSet = tup[2]
                        bestCriteria = tup[3]
                        bestColumn = tup[4]
                        bestGain = currGain

        if bestGain > threshold:
            #Finally split the dataset and create the subtree based on the best values obtained above
            #print "Splitting on Column : "+str(bestColumn)+" with criteria : "+str(bestCriteria)
            '''

            print "Best values : "
            print "Best Gain : "+str(bestGain)
            print "Best Criteria : "+str(bestCriteria)
            print "Best Column : "+str(bestColumn)
            print ""
            '''
            lBranch =  createTree(bestSet[0],depth-1,threshold, cluster = cluster, nominalColumns = nominalColumns, isClassifier = isClassifier)
            rBranch = createTree(bestSet[1],depth-1,threshold, cluster = cluster, nominalColumns = nominalColumns, isClassifier = isClassifier)
            return Node(col = bestColumn, leftBranch = lBranch,rightBranch= rBranch, criteria = bestCriteria)

        else:
            '''
            print ""
            print "No further branching possible "
            print "Adding leaf values : "+str(valuesInColumn)
            '''
            if isClassifier:
                return Node(leafValues= countOccurenceOfClassLabel(subDataSet))
            else:
                regValues = []
                for row in subDataSet:
                    regValues.append(float(row[len(row) - 1]))
                #print regValues
                return Node(leafValues = regValues)

    #No further branching possible since depth has become 0, create a node with all the possible leaf values
    else :
        if isClassifier: 
            return Node(leafValues = countOccurenceOfClassLabel(subDataSet))
        else:
            regValues = []
            for row in subDataSet:
                regValues.append(float(row[len(row) - 1]))
            
            return Node(leafValues = regValues)

'''
The method write result will write the result of the classifier and the expected result in a CSV format.
'''
def writeResult(predictionsPlusExpectedValues, fileName="predictionsWithDepth"):
    with open(fileName,'wb') as f:
        csvWriter = csv.writer(f)
        for row in predictionsPlusExpectedValues:
            csvWriter.writerow(row)
        f.close()


def getRelevantLeafNode(tree, row):
    currentNode = tree

    #Handling the Special case of depth = 0 
    #Recursively searching for the leaf node that martches the criteria
    while(currentNode.leafValues == None):
        if currentNode.col not in nominalColumns:
            #current node is a nominal column..      
            #print currentNode.col, currentNode.criteria
            if float(row[currentNode.col]) <= float(currentNode.criteria): 
                currentNode = currentNode.rightBranch
            else:
                currentNode = currentNode.leftBranch
        else:
            #current node is a continuous column..                    
            if row[currentNode.col] == currentNode.criteria: 
                currentNode = currentNode.rightBranch
            else:
                currentNode = currentNode.leftBranch

    return currentNode

'''
Given a tree and a dataset, the method classifyNewSample will output the predicted classification of each row in the dataset.
'''
def classifyNewSample(tree, testData, fileName, nominalColumns):

    predictionsPlusExpectedValues = []

    for row in testData:
        leaf = getRelevantLeafNode(tree, row).leafValues

        predictedLabel = None
        currentPredictionPlusExpectedValues = []
    
        # Counting the occurences of each possible class label in the leaf
        labelCount = len(leaf)

        #if there is only one label then classify as that label
        if(labelCount == 1):
            predictedLabel = leaf.keys()

        #Else we count the number of occurences of each label and assign the label which has a greater number of occurences
        else:
            probabilityOfClassLabels = {}
            #Counting the total number of occurences of each label
            totalNumberOfLabels = 0
            for key in leaf.keys():
                totalNumberOfLabels += leaf[key]

            #Calculating and assigning the probability of each key to the dictionary probabilityOfClassLabels
            for key in leaf.keys():
                probabilityOfClassLabels[key] = float(leaf[key])/totalNumberOfLabels

            maxProbability = 0.0
            bestKey = None

            '''
            Getting the label with Max Probability, if 2 labels are equally probable then the selection
            depends on the order in which the keys are stored, which is generally random, because the dict in Python stores the dict in an unordered manner 
            2 runs of the program will never have keys in the same order. 
            '''
            for key in leaf.keys():
                if probabilityOfClassLabels[key] > maxProbability:
                    maxProbability = probabilityOfClassLabels[key]
                    bestKey = key
            predictedLabel = bestKey

        #Handles the case where the label is of the type list, this happens when there are multiple labels in one Node
        if(type(predictedLabel) == list):
            currentPredictionPlusExpectedValues.append(str(predictedLabel[0]))
        else: # No issue when there is just one label per node 
            currentPredictionPlusExpectedValues.append(str(predictedLabel))

        #appending the expected result from testData
        currentPredictionPlusExpectedValues.append(row[len(row)-1])
        #List of lists containing the prediction vs expected values
        predictionsPlusExpectedValues.append(currentPredictionPlusExpectedValues)
 
    writeResult(predictionsPlusExpectedValues, fileName)
    return computeStats(predictionsPlusExpectedValues)


def checkDecisionTree(trainingFileName, testFileName, depth=15, isPrintTree=False, nominalColumns = []):
    #Change the trhreshold value if you want to have a minimum information gain at each split, by default we assigned it 0
    threshold=0.0

    trainData = readData(trainingFileName)
    testData = readData(testFileName)
    #isPrintTree = True
    #The variable tree will be an instance of the type Node
    tree = createTree(trainData, depth, nominalColumns = nominalColumns)
    if isPrintTree:
        print ""
        print ""
        print "Structure of the Tree : "
        print ""
        #Printing the tree in a form that helps visualize the structure better
        printTree(tree)
        print ""
        
    #Now that we have the tree built,lets predict output on the test data
    fileName="results/"+"PredictionOf"+testFileName.split('/')[1]
    precision, recall, accuracy = classifyNewSample(tree=tree, testData=testData, fileName=fileName, nominalColumns = nominalColumns)
    #print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    #print "Number of Leaves in the tree is: ", numLeaves(tree)   
    return (1 - computeMisClassfication(fileName), precision, recall, accuracy)

def getClusterValue(row, tree, nominalColumns):
    currentNode = tree
    #Recursively searching for the leaf node that martches the criteria
    while (currentNode.leafValues == None):
        if currentNode.col in nominalColumns:
            # current column is a nominal column
            if row[currentNode.col] == currentNode.criteria: 
                currentNode = currentNode.rightBranch
            else:
                currentNode = currentNode.leftBranch
        else:
            # current column is a not nominal column
            if float(row[currentNode.col]) <= float(currentNode.criteria): 
                currentNode = currentNode.rightBranch
            else:
                currentNode = currentNode.leftBranch       
    return currentNode.clusterNum


def newLogic(train, test, priv_train, priv_depth, privNominalColumns, prunedNominalColumns):
    global numClusters
    nodes = []    
    #construct a tree using priv information..
    
    #privTree = createTree(readData(priv_train), priv_depth, nominalColumns = nominalColumns)
    privTree = createTree(readData(priv_train), nominalColumns = privNominalColumns) #TODO: check the use of the priv_depth..
    #assign a new cluster number to each leaf node..
    index = 0
    nodes.append(privTree)
    while len(nodes) != 0:
        node = nodes.pop()
        if node.leftBranch == None and node.rightBranch == None and node.clusterNum == 0:
            node.clusterNum = index
            index += 1
        if node.leftBranch != None:
            nodes.append(node.leftBranch)
        if node.rightBranch != None:
            nodes.append(node.rightBranch)
    numClusters = index
    #print "Total clusters is: ", index
    trainData = readData(train)
    testData = readData(test)
    privData = readData(priv_train)
    cluster = {} #clear the previous cluster global variable..
    numRows = len(trainData)
    for i in range(numRows):
        cluster[",".join(trainData[i])] = getClusterValue(privData[i], privTree, privNominalColumns)
        #print cluster[",".join(trainData[i])]

    threshold = 0
    tree = createTree(trainData, 15, threshold, True, cluster = cluster, nominalColumns = prunedNominalColumns)

    #Now that we have the tree built,lets predict output on the test data
    fileName="results/"+"PredictionOf"+test.split('/')[1]
    precision, recall, accuracy = classifyNewSample(tree=tree, testData=testData, fileName=fileName, nominalColumns = prunedNominalColumns)
    #print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    #print "Number of Leaves in the tree is: ", numLeaves(tree)   
    return (1 - computeMisClassfication(fileName), precision, recall, accuracy)        

def combineGain(normalGain, privGain, isClassifier):
    global alpha

    if isClassifier:
        #print normalGain, privGain
        #return privGain
        privGain = alpha * privGain
        #return privGain + normalGain
        #return normalGain
        #
        #'''
        if normalGain < privGain:
            return reverseHarmonicMean(normalGain, privGain)
        else:
            return harmonicMean(normalGain, privGain)
        #'''
        return reverseHarmonicMean(normalGain, privGain)
        #return harmonicMean(normalGain, privGain)
    else:
        #TODO: replace this logic with a new logic..
        privGain = alpha * privGain
        #return normalGain
        return privGain + normalGain        
    #'''
    #old logic..
    '''
    if normalGain < privGain:
        return reverseHarmonicMean(normalGain, privGain)
    else:
        return harmonicMean(normalGain, privGain)
    #'''
    
    '''
    #Semi harmonic and linear logic..
    if normalGain > privGain:
        return (normalGain+privGain)/2.0
    else:
        return harmonicMean(normalGain, privGain)
    '''

    '''
    if normalGain > privGain:
        return reverseHarmonicMean(normalGain, privGain)
    else:
        return harmonicMean(normalGain, privGain)
    '''

#The main function that calls all other functions, execution begins here
def main():

    global prunedNominalColumns
    global privNominalColumns

    global datasets
    global alpha
    global totalParts
    global splitCount

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
                
                #if part != 1:
                    #continue #TODO: remove this..
                #'''
                #print ""
                #print "#"*40
                print "Running "+datasetName+" with fold: ", part
                
                #print "\nRunning entire dataset"
                #checkDecisionTree(datasetName+"/"+dirName+"/complete_train_"+str(part)+".csv", datasetName+"/"+dirName+"/complete_test_"+str(part)+".csv")
                
                #print "\nRunning only privileged information"
                #privAccHolder, precision, recall, accuracy = checkDecisionTree(datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv", datasetName+"/"+dirName+"/priv_test_"+str(part)+".csv")
                #privAcc += privAccHolder
                            
                #print "\nRunning only privileged information with max depth = 3"
                #checkDecisionTree(datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv", datasetName+"/"+dirName+"/priv_test_"+str(part)+".csv", 3, False)
                
                #'''
                print "\nRunning pruned dataset"
                currNormalAcc, precision, recall, accuracy = checkDecisionTree(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv", datasetName+"/"+dirName+"/pruned_test_"+str(part)+".csv", nominalColumns = prunedNominalColumns[datasetName])
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


                #print currNormalAcc," ##" 
                newAcc.append([])
                newPrecision.append([])
                newRecall.append([])
                for run in range(20):
                    alpha = (run+1)/10.0
                    print "Running the new logic with alpha = ",alpha
                    currAcc, precision, recall, accuracy = newLogic(datasetName+"/"+dirName+"/pruned_train_"+str(part)+".csv", datasetName+"/"+dirName+"/pruned_test_"+str(part)+".csv", datasetName+"/"+dirName+"/priv_train_"+str(part)+".csv", 3, privNominalColumns = privNominalColumns[datasetName], prunedNominalColumns = prunedNominalColumns[datasetName])     
                    print currAcc,"\t",alpha
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
