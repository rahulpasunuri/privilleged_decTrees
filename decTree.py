#!/usr/bin/python
import sys
import csv
from math import *
from Node import Node
from computeStats import *

'''
The method readData, given a CSV file name, reads the data and returns the data set as a list of lists.
Each element in the list is a list.
'''
def readData(fileName):
    data = []
    myFile = open(fileName, 'rt')
    try:
        reader = csv.reader(myFile)
        for row in reader:
            data.append(row)
    except:
        print "Error opening File: "+fileName
        exit()
    finally:
        myFile.close()
    return data

'''
The method calcEntropy takes a dataset as input and returns its entropy calculated on the basis of 
the number of occurences of each class label.
Here is exactly where the use of the method countOccurenceOfClassLabel() comes into play.
'''
def calcEntropy(subDataSet):
    classLablelCounts = countOccurenceOfClassLabel(subDataSet)
    totalRows = len(subDataSet)
    entropy = 0.0

    for key in classLablelCounts:
        p = float(classLablelCounts[key])/totalRows
        entropy -= p*log(p,2)

    return entropy

cluster = {}

def calcPrivEntropy(data):
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

def combineGain(normalGain, privGain):
    global alpha
    
    privGain = alpha * privGain
    #return normalGain
    return privGain + normalGain

    '''
    #old logic..
    if normalGain < privGain:
        return reverseHarmonicMean(normalGain, privGain)
    else:
        return harmonicMean(normalGain, privGain)
    '''
    
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
'''
The createTree function is where all the magic happens, 
We call createTree recursively until we reach the required depth or a good decision tree
The method takes a sub part of the dataset as input and creates a tree based on the decision criteria.

'''
def createTree(subDataSet, depth=15,threshold=0.0, isPrivAvailable = False):

    #Counting the number of rows in the Dataset
    numOfRows = len(subDataSet)

    #if the required depth is > 0 and the dataset has some rows 
    if depth > 0 and len(subDataSet) > 0:
        '''
        print "Current Depth : "+str(depth)
        print ""
        '''
        #We first calculate the entropy for the entire data set
        entropy = calcEntropy(subDataSet)

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
            #We are now iterating through each value in the current iteration of column to see which value serves as the best split
            for value in valuesInColumn:
                #Split the dataset on the current value of column and value
                (set1,set2) = splitData(subDataSet,col, value)
                if len(set1) > 0 and len(set2) > 0:
                    #Calculate infoGain for each col and each value in the column
                    if isPrivAvailable == False:
                        infoGain = calcInfoGain(entropy, set1,set2)
                        #Choose the best col and value 
                        if infoGain > bestGain and len(set1) > 0 and len(set2) > 0 :
                            bestGain = infoGain
                            bestSet = (set1, set2)
                            bestCriteria = value
                            bestColumn = col
                    else:
                        currGain, privGain = calcPrivInfoGain(entropy, calcPrivEntropy(subDataSet), set1,set2)
                        privGainList.append((currGain, privGain, (set1, set2), value, col))
                        normalAvg += currGain
                        privAvg += privGain
                
            if isPrivAvailable == True:
                #'''
                normalAvg = normalAvg/len(privGainList)
                privAvg = privAvg/len(privGainList)
                shift = abs(normalAvg - privAvg)
                #shift = 0 #TODO: check the usage of shift..
                if privAvg > normalAvg:
                    index = 0
                else:
                    index = 1
                
                for ind in range(len(privGainList)):
                    currTuple = privGainList[ind]
                    if index == 0:
                        finalTuple = (currTuple[0]+ abs(shift), currTuple[1], currTuple[2], currTuple[3], currTuple[4])
                    else:
                        finalTuple = (currTuple[0], currTuple[1]+ abs(shift), currTuple[2], currTuple[3], currTuple[4])
                    privGainList[ind] = finalTuple
                #'''
                #find the max threshold
                for tup in privGainList:
                    currGain = combineGain(tup[0], tup[1])
                    if currGain > bestGain and len(tup[2][0]) > 0 and len(tup[2][1]) > 0:
                        #print tup[0],"\t",tup[1],"\t", currGain
                        bestSet = tup[2]
                        bestCriteria = tup[3]
                        bestColumn = tup[4]
                        bestGain = currGain

        if bestGain > threshold:
            #Finally split the dataset and create the subtree based on the best values obtained above
            '''
            print "Splitting on Column : "+str(bestColumn)+" with criteria : "+str(bestCriteria)

            print "Best values : "
            print "Best Gain : "+str(bestGain)
            print "Best Criteria : "+str(bestCriteria)
            print "Best Column : "+str(bestColumn)
            print ""
            '''
            lBranch =  createTree(bestSet[0],depth-1,threshold)
            rBranch = createTree(bestSet[1],depth-1,threshold)
            return Node(col = bestColumn, leftBranch = lBranch,rightBranch= rBranch, criteria = bestCriteria)

        else:
            '''
            print ""
            print "No further branching possible "
            print "Adding leaf values : "+str(valuesInColumn)
            '''
            return Node(leafValues= countOccurenceOfClassLabel(subDataSet))

    #No further branching possible since depth has become 0, create a node with all the possible leaf values
    else : 
        return Node(leafValues = countOccurenceOfClassLabel(subDataSet))

'''
The method calcInfoGain returns the Information Gain when passed with the current value of entropy, and dataset split on a particular value of a particular column.
This used to find which is the best column to split the dataset on and subsequently decide what should the criteria be. 
'''
def calcInfoGain(currentEntropy, subDataSet1,subDataSet2):
    p = float(len(subDataSet1))/(len(subDataSet1)+len(subDataSet2))
    infoGain = currentEntropy - p*calcEntropy(subDataSet1) - (1-p)*calcEntropy(subDataSet2)
    return infoGain

def harmonicMean(a1, a2):
    if a1*a2 == 0:
        return 0
    else:
        return (2*a1*a2)/(a1+a2)     

def geoMean(a1, a2):
    if a1*a2 == 0:
        return 0
    else:
        return sqrt(a1*a2)     

def reverseHarmonicMean(a1, a2):
    if a1+a2 == 0:
        return 0
    else:
        return (a1*a1 + a2*a2)/(a1+a2)
        
def calcPrivInfoGain(currentEntropy, clusterEntropy, subDataSet1,subDataSet2):
    global numClusters
    global alpha
    normalGain = calcInfoGain(currentEntropy, subDataSet1, subDataSet2)
    p = float(len(subDataSet1))/(len(subDataSet1)+len(subDataSet2))
    privGain = clusterEntropy -p*calcPrivEntropy(subDataSet1) - (1-p)*calcPrivEntropy(subDataSet2)
    
    #privGain = log(numClusters,2)*privGain #TODO: check this 
    #privGain = 1.0/numClusters*privGain #TODO: check this 
    #print normalGain, privGain
    #return  geoMean(normalGain, privGain)
    #return normalGain + privGain
    #return min(normalGain, privGain)
    #return privGain
    #return max(normalGain, privGain)
    #return privGain
    #return harmonicMean(normalGain, privGain)
    #ratio =  normalGain/privGain
    #print alpha
    #return alpha*normalGain + privGain
    #print ratio
    return (normalGain, privGain)
    '''
    if normalGain < privGain:
        #return (normalGain + privGain)/2
        #return geoMean(normalGain, privGain)
        return reverseHarmonicMean(normalGain, privGain)
        #return harmonicMean(normalGain, privGain)
    else:
        #return (normalGain + privGain)/2
        return harmonicMean(normalGain, privGain)
    #return harmonicMean(normalGain, privGain)
    #
    '''
'''
The method countOccurenceOfClassLabel is called whenever we need to count how many times each class label occurs in a the subDataSet. 
This will be used to calculate Entropy and Infogain
It returns a dictionary that has keys as the class label and the values as the number of Occurences of that class label
'''
def countOccurenceOfClassLabel(subDataSet):
    countsOfLabels = {}
    for row in subDataSet:
        if row[len(row)-1] in countsOfLabels : 
            countsOfLabels[row[len(row)-1]] += 1    
        else :
            countsOfLabels[row[len(row)-1]] = 1
    return countsOfLabels

'''
The method printTree takes a tree of the type Node and an indent value. It outputs the tree in a human interpretable form 
by showing subsequent branches with indents. 
'''
def printTree(tree, indent=''):
    if tree.leafValues != None:
        print "Leaf Node : "+str(tree.leafValues)
    else:
        print "Split on Column : "+str(tree.col)+" with criteria : "+str(tree.criteria)
        print indent+"Left Branch -> ",
        printTree(tree.leftBranch,indent="     "+indent)
        print indent+"Right Branch -> ",
        printTree(tree.rightBranch,indent="     "+indent)

'''
The method write result will write the result of the classifier and the expected result in a CSV format.
'''
def writeResult(predictionsPlusExpectedValues,depth="",fileName="predictionsWithDepth"):
    with open(fileName,'wb') as f:
        csvWriter = csv.writer(f)
        for row in predictionsPlusExpectedValues:
            csvWriter.writerow(row)
        f.close()

'''
Given a tree and a dataset, the method classifyNewSample will output the predicted classification of each row in the dataset.
'''
def classifyNewSample(tree, testData,depth,fileName):
	
	predictionsPlusExpectedValues = []

	for row in testData:

		currentNode = tree
		leaf = None
		predictedLabel = None
		currentPredictionPlusExpectedValues = []

		#Handling the Special case of depth = 0 
		if(depth == 0):
			leaf = tree.leafValues
		else:
			#Recursively searching for the leaf node that martches the criteria
			while(leaf == None):
				if float(row[currentNode.col]) <= float(currentNode.criteria): 
					currentNode = currentNode.rightBranch
				else:
					currentNode = currentNode.leftBranch
				leaf = currentNode.leafValues

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
	 
	writeResult(predictionsPlusExpectedValues, depth, fileName)
	return computeStats(predictionsPlusExpectedValues)

'''
The method splitData takes a dataset as input and splits it into 2 based on the criteria on the specified column and returns the resulting 2 datasets.
Provide Column value as if counting from ZERO.
'''	
def splitData(subDataSet, column, criteria):
	subDataSet1=[] #All samples that match the criteria
	subDataSet2=[] #All samples that do not match the criteria
	for row in subDataSet:
		#Doing a <= and > split..
		if(float(row[column]) <= float(criteria)):
			subDataSet1.append(row) 
		else:
			subDataSet2.append(row)
	return (subDataSet2,subDataSet1)

def computeMisClassfication(filename):
    f = open(filename,"r")
    lines=f.readlines()
    f.close()

    totalCount=len(lines)
    misClassificationCount=0
    for t in lines:
        words = t.strip().split(',')
        if(words[0]!=words[1]):
            misClassificationCount+=1
    rate = float(misClassificationCount)/totalCount
    return rate

def numLeaves(tree):
    if tree.leftBranch == None and tree.rightBranch == None:
        return 1
    elif tree == None:
        return 0
    else:
        return numLeaves(tree.leftBranch) + numLeaves(tree.rightBranch)


def checkDecisionTree(trainingFileName, testFileName, depth=15, isPrintTree=False):
    #Change the trhreshold value if you want to have a minimum information gain at each split, by default we assigned it 0
    threshold=0.0

    trainData = readData(trainingFileName)
    testData = readData(testFileName)
    #isPrintTree = True
    #The variable tree will be an instance of the type Node
    tree = createTree(trainData, depth)
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
    precision, recall, accuracy = classifyNewSample(tree=tree, testData=testData,depth=depth,fileName=fileName)
    #print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    #print "Number of Leaves in the tree is: ", numLeaves(tree)   
    return (1 - computeMisClassfication(fileName), precision, recall, accuracy)

def getClusterValue(row, tree):
    currentNode = tree
    #Recursively searching for the leaf node that martches the criteria
    while (currentNode.leafValues == None):
        if float(row[currentNode.col]) <= float(currentNode.criteria): 
            currentNode = currentNode.rightBranch
        else:
            currentNode = currentNode.leftBranch
    return currentNode.clusterNum

numClusters = 0

def newLogic(train, test, priv_train, priv_depth):
    global cluster
    global numClusters
    nodes = []    
    #construct a tree using priv information..
    
    #privTree = createTree(readData(priv_train), priv_depth)
    privTree = createTree(readData(priv_train)) #TODO: check the use of the priv_depth..
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
    cluster = {}
    numRows = len(trainData)
    for i in range(numRows):
        cluster[",".join(trainData[i])] = getClusterValue(privData[i], privTree)
        #print cluster[",".join(trainData[i])]
    threshold = 0
    tree = createTree(trainData, 15, threshold, True)
    '''
    print ""
    print ""
    print "Structure of the Tree : "
    print ""
    #Printing the tree in a form that helps visualize the structure better
    
    print ""
    '''
    #printTree(tree)
    #Now that we have the tree built,lets predict output on the test data
    fileName="results/"+"PredictionOf"+test.split('/')[1]
    precision, recall, accuracy = classifyNewSample(tree=tree, testData=testData,depth=15,fileName=fileName)
    #print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    #print "Number of Leaves in the tree is: ", numLeaves(tree)   
    return (1 - computeMisClassfication(fileName), precision, recall, accuracy)        
alpha = 0            
datasets = []
datasets.append("random")
datasets.append("heart")
datasets.append("breast")
datasets.append("heart_multi")
datasets.append("iris")
datasets.append("diabetes")
#The main function that calls all other functions, execution begins here
def main():
    global alpha
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
        
        for part in range(5):

            #if part != 1:
                #continue #TODO: remove this..
            #'''
            #print ""
            #print "#"*40
            #print "Running "+datasetName+" with fold: ", part
            
            #print "\nRunning entire dataset"
            #checkDecisionTree(datasetName+"/complete_train_"+str(part)+".csv", datasetName+"/complete_test_"+str(part)+".csv")
            
            #print "\nRunning only privileged information"
            #privAccHolder, precision, recall, accuracy = checkDecisionTree(datasetName+"/priv_train_"+str(part)+".csv", datasetName+"/priv_test_"+str(part)+".csv")
            #privAcc += privAccHolder
                        
            #print "\nRunning only privileged information with max depth = 3"
            #checkDecisionTree(datasetName+"/priv_train_"+str(part)+".csv", datasetName+"/priv_test_"+str(part)+".csv", 3, False)
            
            #'''
            print "\nRunning pruned dataset"
            currNormalAcc, precision, recall, accuracy = checkDecisionTree(datasetName+"/pruned_train_"+str(part)+".csv", datasetName+"/pruned_test_"+str(part)+".csv")
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
            for run in range(1, 21):
                alpha = run/10.0
                print "Running the new logic with alpha = ",alpha
                currAcc, precision, recall, accuracy = newLogic(datasetName+"/pruned_train_"+str(part)+".csv", datasetName+"/pruned_test_"+str(part)+".csv", datasetName+"/priv_train_"+str(part)+".csv", 3)     
                #print currAcc,"\t",alpha
                newAcc[part].append(currAcc)
                newPrecision[part].append(precision)
                newRecall[part].append(recall)
            #print "#"*40
            #print ""
        print "\nNormal Accuracy is", normalAcc/5.0
        #print "Privileged Accuracy is", privAcc/5.0
        for label in normalPrecision:
            print "Stats for label: ",label
            print "\tPrecision is: ", normalPrecision[label]/5.0
            print "\tRecall is: ", normalRecall[label]/5.0

            print "-"*30

        avgAcc = [0 for i in range(20)]
        avgPrecision = [ {} for i in range(20)]
        avgRecall = [ {} for i in range(20) ]
        for run in range(20):
            for j in range(5):
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

        print "\nNew Accuracy is: ",maxAvgAccuracy/5.0, "for alpha: ",maxAlpha
        for lbl in normalAccuracy:
            print "Stats for label: ",label
            print "\tPrecision for the chosen alpha is: ", avgPrecision[chosenI][lbl]/5.0
            print "\tRecall for the chosen alpha is: ", avgRecall[chosenI][lbl]/5.0
            print "-"*30
#Execution begins here
if __name__ == "__main__" : main()
