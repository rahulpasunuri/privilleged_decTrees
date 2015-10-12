#!/usr/bin/python
import sys
import csv
from math import *
from Node import Node

'''
This program knows no difference between a number or a string, It considers everything as a string. (Either a string matches the criteria or it doesn't) 
This makes it a Binary Tree. Ultimately entropy and informatin gain depends on the probability of a certain value.
It does not matter what the value is .. 
I think this will work in most cases .. Lets see ... 
'''

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

            #We are now iterating through each value in the current iteration of column to see which value serves as the best split
            for value in valuesInColumn:

                #Split the dataset on the current value of column and value
                (set1,set2) = splitData(subDataSet,col, value)
                if len(set1) > 0 and len(set2) > 0:
                    #Calculate infoGain for each col and each value in the column
                    if isPrivAvailable == False:
                        infoGain = calcInfoGain(entropy, set1,set2)
                    else:
                        infoGain = calcPrivInfoGain(entropy, calcPrivEntropy(subDataSet), set1,set2)
                    #Choose the best col and value 
                    if infoGain > bestGain and len(set1) > 0 and len(set2) > 0 :
                        bestGain = infoGain
                        bestSet = (set1, set2)
                        bestCriteria = value
                        bestColumn = col

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
    normalGain = calcInfoGain(currentEntropy, subDataSet1, subDataSet2)
    p = float(len(subDataSet1))/(len(subDataSet1)+len(subDataSet2))
    privGain = clusterEntropy -p*calcPrivEntropy(subDataSet1) - (1-p)*calcPrivEntropy(subDataSet2)
    
    #privGain = log(numClusters,2)*privGain #TODO: check this 
    #privGain = 1.0/numClusters*privGain #TODO: check this 
    #print normalGain, privGain
    #return  geoMean(normalGain, privGain)
    #return normalGain + privGain
    #return min(normalGain, privGain)
    #return normalGain
    #return max(normalGain, privGain)
    #return privGain
    #return harmonicMean(normalGain, privGain)
    ratio =  normalGain/privGain
    #print ratio
    if normalGain > privGain:
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
    classifyNewSample(tree=tree, testData=testData,depth=depth,fileName=fileName)
    print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    print "Number of Leaves in the tree is: ", numLeaves(tree)    

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
    
    privTree = createTree(readData(priv_train), priv_depth)
    #privTree = createTree(readData(priv_train)) #TODO: check the use of the priv_depth..
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
    print "Total clusters is: ", index
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
    printTree(tree)
    print ""
    '''
    #Now that we have the tree built,lets predict output on the test data
    fileName="results/"+"PredictionOf"+test.split('/')[1]
    classifyNewSample(tree=tree, testData=testData,depth=15,fileName=fileName)
    print "Accuracy is: ",(1 - computeMisClassfication(fileName))
    print "Number of Leaves in the tree is: ", numLeaves(tree)   
            
datasets = []
#datasets.append("heart")
datasets.append("breast")
#datasets.append("heart_multi")
#The main function that calls all other functions, execution begins here
def main():
    for datasetName in datasets:
        for part in range(5):
            #if part != 1:
                #continue #TODO: remove this..
            print ""
            print "#"*40
            print "Running "+datasetName+" with fold: ", part
            #'''
            print "\nRunning entire dataset"
            checkDecisionTree(datasetName+"/complete_train_"+str(part)+".csv", datasetName+"/complete_test_"+str(part)+".csv")
            
            print "\nRunning only privileged information"
            checkDecisionTree(datasetName+"/priv_train_"+str(part)+".csv", datasetName+"/priv_test_"+str(part)+".csv")

            print "\nRunning only privileged information with max depth = 3"
            checkDecisionTree(datasetName+"/priv_train_"+str(part)+".csv", datasetName+"/priv_test_"+str(part)+".csv", 3, False)
            #'''

            print "\nRunning pruned dataset"
            checkDecisionTree(datasetName+"/pruned_train_"+str(part)+".csv", datasetName+"/pruned_test_"+str(part)+".csv")

            print "\nRunning the new logic.."
            newLogic(datasetName+"/pruned_train_"+str(part)+".csv", datasetName+"/pruned_test_"+str(part)+".csv", datasetName+"/priv_train_"+str(part)+".csv", 3)

            print "#"*40
            print ""
#Execution begins here
if __name__ == "__main__" : main()
