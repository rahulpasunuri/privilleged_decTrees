import sys
import csv
from math import *
from Node import Node
import numpy
from computeStats import *
#list of parameters

limitGainBounds = True


datasets = []
#datasets.append("random")
#datasets.append("heart")
#datasets.append("breast")
#datasets.append("heart_multi")
datasets.append("iris")
#datasets.append("diabetes")

#----new datasets (yet to complete) ---
#datasets.append("glass_binary")
#datasets.append("car") #TODO: no difference with any kind of splits...
#datasets.append("ecoli_binary")

#datasets.append("census") #TODO: takes a lot of time..
#datasets.append("credit")
#datasets.append("hepatitis") #TODO: has a lot of missing values.. how to support them ???


classLabels = {}
#TODO: check whether the below labels are assigned correctly or not..
classLabels["heart"] = [0, 1]

classLabels["breast"] = [0, 1]

classLabels["random"] = [0, 1]

classLabels["iris"] = [0, 1, 2]

classLabels["diabetes"] = [0, 1]

classLabels["glass_binary"] = [0, 1]

classLabels["car"] = [0, 1]

classLabels["census"] = [0, 1]

classLabels["credit"] = [0, 1]

classLabels["ecoli_binary"] = [0, 1]

classLabels["hepatitis"] = [0, 1]




splitCount = 5
totalParts = 5
numClusters = 0
alpha = 0

privilegedColumns = {}

#below two parameters have to be set for every dataset..

#privilegedColumns["heart"] = [1, 2, 3, 4, 5, 6, 7, 8]
privilegedColumns["heart"] = [1, 3, 4, 6, 9, 10, 12]

#below three give good results..
#privilegedColumns["heart"] = [1, 2, 3, 11, 12]

#privilegedColumns["heart"] = [2, 3, 4, 6, 9, 11, 12]

#privilegedColumns["heart"] = [1, 5, 7, 8, 10, 13]
#privilegedColumns["heart"] = [1, 3, 4, 6, 9]

#privilegedColumns["heart"] = [1, 4,9, 10, 11, 12]

privilegedColumns["heart_multi"] = [1, 2, 3, 6, 9, 10, 12]

#privilegedColumns["breast"] = [1, 2, 7, 12, 13, 14, 24, 28] #OLD
privilegedColumns["breast"] = [1, 2, 7, 12, 13, 14, 22, 23, 24, 25, 27, 28]

privilegedColumns["random"] = [1, 3, 5, 7, 8]

privilegedColumns["iris"] = [2,3]

#privilegedColumns["diabetes"] = [0,2,4,5]
#privilegedColumns["diabetes"] = [1,2, 5, 7]
privilegedColumns["diabetes"] = [0, 3, 4, 6]

privilegedColumns["glass_binary"] = [5, 6, 7, 8]

#TODO: get privileged columns for the car dataset..
privilegedColumns["car"] = [ 3]

#TODO: get privileged columns for the census dataset..
privilegedColumns["census"] = [1,2,3]

privilegedColumns["credit"] = [1, 3, 6, 8, 9, 13, 10] # reduces acc..

privilegedColumns["ecoli_binary"] = [1, 2, 3]

#TODO: get privileged columns for the hepatitis dataset..
privilegedColumns["hepatitis"] = [1,2,3]


#lists all the nominal columns in every dataset..
#NOTE: ******The columns in the below variable must be ordered in an ascending order..
nominalColumns = {}
nominalColumns["heart"] = [] #on nominal columns
nominalColumns["heart_multi"] = [] #on nominal columns
nominalColumns["breast"] = [] #on nominal columns
nominalColumns["iris"] = [] #on nominal columns
nominalColumns["diabetes"] = [] #on nominal columns
nominalColumns["glass_binary"] = [] #on nominal columns
nominalColumns["car"] = [0, 1, 4, 5] #on nominal columns
nominalColumns["census"] = [1,3,5,6,7,8,9,13] #on nominal columns
nominalColumns["credit"] = [0, 3,4,5,6,8,9,11,12] #on nominal columns
nominalColumns["ecoli_binary"] = [] #on nominal columns

prunedNominalColumns = {}
privNominalColumns = {}

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
The method calcInfoGain returns the Information Gain when passed with the current value of entropy, and dataset split on a particular value of a particular column.
This used to find which is the best column to split the dataset on and subsequently decide what should the criteria be. 
'''
def calcInfoGain(currentEntropy, subDataSet1, subDataSet2, isClassifier):
    len1 = len(subDataSet1)
    len2 = len(subDataSet2)
    totalLen = len1 + len2

    p = float(len1)/totalLen

    if isClassifier:
        infoGain = currentEntropy - p*calcEntropy(subDataSet1) - (1-p)*calcEntropy(subDataSet2)
    else:
        infoGain = currentEntropy*totalLen - len1*calcVariance(subDataSet1) - len2*calcVariance(subDataSet2)
    
    return infoGain

def harmonicMean(a1, a2):
    if a1+a2 == 0:
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
The method splitData takes a dataset as input and splits it into 2 based on the criteria on the specified column and returns the resulting 2 datasets.
Provide Column value as if counting from ZERO.
'''	
def splitData(subDataSet, column, criteria, nominalColumns):
	subDataSet1=[] #All samples that match the criteria
	subDataSet2=[] #All samples that do not match the criteria
	for row in subDataSet:
		#Doing a <= and > split..
		if column not in nominalColumns:
		    if float(row[column]) <= float(criteria):
			    subDataSet1.append(row) 
		    else:
			    subDataSet2.append(row)
		else:
		    if row[column].strip() == criteria.strip():
			    subDataSet1.append(row) 
		    else:
			    subDataSet2.append(row)
	return (subDataSet2,subDataSet1)
	
	
def numLeaves(tree):
    if tree.leftBranch == None and tree.rightBranch == None:
        return 1
    elif tree == None:
        return 0
    else:
        return numLeaves(tree.leftBranch) + numLeaves(tree.rightBranch)
        
'''
The method countOccurenceOfClassLabel is called whenever we need to count how many times each class label occurs in a the subDataSet. 
This will be used to calculate Entropy and Infogain
It returns a dictionary that has keys as the class label and the values as the number of Occurences of that class label
'''
def countOccurenceOfClassLabel(subDataSet):
    countsOfLabels = {}
    for row in subDataSet:
        if row[len(row)-1] in countsOfLabels: 
            countsOfLabels[row[len(row)-1]] += 1    
        else:
            countsOfLabels[row[len(row)-1]] = 1
    return countsOfLabels


def getSplitName(num):
    return "split"+str(num)
     
'''
The method calcEntropy takes a dataset as input and returns its entropy calculated on the basis of 
the number of occurences of each class label.
Here is exactly where the use of the method countOccurenceOfClassLabel() comes into play.
'''
def calcEntropy(subDataSet):
    totalRows = len(subDataSet)
    entropy = 0.0
    
    classLablelCounts = countOccurenceOfClassLabel(subDataSet)
    for key in classLablelCounts:
        p = float(classLablelCounts[key])/totalRows
        entropy -= p*log(p,2)

    return entropy

def calcVariance(subDataSet):
    vals = []
    for row in subDataSet:
        vals.append(float(row[len(row)-1]))
    return numpy.var(vals)
    
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

def init():
    computeFinalNominalColumns()
    
def computeFinalNominalColumns():
    global prunedNominalColumns
    global privNominalColumns
    
    for datasetName in datasets:
        #initialize the global variables..
        prunedNominalColumns[datasetName] = []
        privNominalColumns[datasetName] = []
    
        origNominalCols = nominalColumns[datasetName]
        privColumns = privilegedColumns[datasetName]

        '''
        print origNominalCols
        print privColumns
        '''
        
        if len(origNominalCols) == 0:
            continue # no nominal columns in the dataset..
        
        max_orig_nominal = origNominalCols[len(origNominalCols) - 1]
        prunedColCount = 0
        privColCount = 0
        
        for col_num in range(max_orig_nominal+1):
            if col_num in privColumns:
                if col_num in origNominalCols:
                    privNominalColumns[datasetName].append(privColCount)
                privColCount += 1
            else: # this column is not a privileged column..
                if col_num in origNominalCols:
                    prunedNominalColumns[datasetName].append(prunedColCount)
                prunedColCount += 1  
    '''
    print prunedNominalColumns
    print privNominalColumns
    exit()
    '''
    
def getTreeDepth(tree):
    if tree == None:
        return 0

    if tree.leftBranch == None and tree.leftBranch == None:
        return 1
    return max(getTreeDepth(tree.leftBranch), getTreeDepth(tree.rightBranch)) + 1

