import sys
import csv
import random
import decTree

isSimplificationEnabled = False
isSplitEnabled = False
isPrivSplitEnabled = True

if isSimplificationEnabled:
    ##############################PART-1##############################
    #replace all the non-zero labels to zero..
    print "Simplifying the dataset"
    f = open("original/heart.csv", "r")
    out = open("original/heart_simplified.csv", "w")
    lines = f.readlines()
    for l in lines:
        words = l.strip().split(",")
        classIndex = len(words) - 1 
        if '?' in l:
            #ignore the rows with missing values..
            continue
            
        if words[classIndex] != "0":
             words[classIndex] = "1"    
        out.write(",".join(words)+"\n")
    out.close()

if isSplitEnabled:
    ##############################PART-II##############################
    #split the dataset to train and test....
    print "Splitting the dataset to test and train"
    entireDataSet = []
    trainData = []
    testData = []

    #Getting the dataset from CSV file to memory
    #Reusing the method readData from decTree.py
    inp = open("original/heart_simplified.csv", "r")
    entireDataSet = inp.readlines()
    numberOfLines=len(entireDataSet)
    random.seed()
    random.shuffle(entireDataSet)
    finalValue = 0
    for x in range(numberOfLines):
        if x > 0.67*numberOfLines:
            break
        trainData.append(entireDataSet[x])

    for x in range(numberOfLines):
        if x < 0.67*numberOfLines:
            continue
        testData.append(entireDataSet[x])

    train = open("original/heart_train.csv", "w")
    test = open("original/heart_test.csv", "w")
    for l in trainData:
        train.write(l)
    train.close()

    for l in testData:
        test.write(l)
    test.close()

def split(orig, pruned, priv):
    for l in orig:
        l=l.strip()
        words = l.split(",")
        privColumns = [1, 4, 9, 10, 11, 12]
        privInfo = []
        pruneInfo = []
        for i in range(len(words)):
            if i in privColumns:
                privInfo.append(words[i])
            else:
                pruneInfo.append(words[i])
        privInfo.append(words[len(words) -1])
        priv.write(",".join(privInfo)+"\n")
        pruneInfo.append(words[i])
        pruned.write(",".join(pruneInfo)+"\n")


if isPrivSplitEnabled:
    ##############################PART-III##############################
    #create the privilleged datasets..
    print "Splitting the privilleged information!!"
    train = open("original/heart_train.csv", "r").readlines()
    test = open("original/heart_test.csv", "r").readlines()

    priv_train = open("original/heart_priv_train.csv", "w")
    pruned_train = open("original/heart_pruned_train.csv", "w")

    priv_test = open("original/heart_priv_test.csv", "w")
    pruned_test = open("original/heart_pruned_test.csv", "w")

            
    split(train, pruned_train, priv_train,)
    split(test, pruned_test, priv_test)

