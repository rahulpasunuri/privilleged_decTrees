import sys
import csv
import random
import decTree
import os
from globalConstants import * #file containing all the global constants..

isSimplificationEnabled = True
isSplitEnabled = True
isPrivSplitEnabled = True

def split(orig, pruned, priv, privColumns):
    global privilegedColumns
    global datasets

    for l in orig:
        l=l.strip()
        words = l.split(",")
        privInfo = []
        pruneInfo = []
        for i in range(len(words)):
            if i in privColumns:
                privInfo.append(words[i])
            else:
                pruneInfo.append(words[i])
        privInfo.append(words[len(words) -1])
        priv.write(",".join(privInfo)+"\n")
        pruned.write(",".join(pruneInfo)+"\n")

for datasetName in datasets:
    if isSimplificationEnabled:
        ##############################PART-1##############################
        #replace all the non-zero labels to zero..
        print "Simplifying the dataset"
        f = open(datasetName+"/dataset.csv", "r")
        out = open(datasetName+"/dataset_simplified.csv", "w")
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l == "":
                continue # skip empty lines..
            if datasetName == "heart":
                words = l.strip().split(",")
                classIndex = len(words) - 1 
                if '?' in l:
                    #ignore the rows with missing values..
                    continue
                    
                if words[classIndex] != "0":
                     words[classIndex] = "1"    
            elif datasetName == "breast":
                cols = l.strip().split(",")
                classIndex = 1
                words = []
                for colNum in range(len(cols)):
                    if colNum != classIndex:
                        words.append(cols[colNum])
                if cols[classIndex].lower() == "m":
                    words.append("1") #malignant..
                else:
                    words.append("0") #benign case..
            elif datasetName == "heart_multi":
                words = l.strip().split(",")
                classIndex = len(words) - 1 
                if '?' in l:
                    #ignore the rows with missing values..
                    continue

            elif datasetName == "iris":
                words = l.strip().split(",")
                classIndex = len(words) - 1
                if "setosa" in words[classIndex]:
                    words[classIndex] = '0'
                elif "virginica" in words[classIndex]:
                    words[classIndex] = '1'
                elif "versicolor" in words[classIndex]:
                    words[classIndex] = '2'
            else:
                words = l.strip().split(",")
                classIndex = len(words) - 1 
                
            out.write(",".join(words)+"\n")
        out.close()

    random.seed()
    for split_num in range(splitCount):
        dirName = getSplitName(split_num)
        if not os.path.exists(datasetName+"/"+dirName):
            os.makedirs(datasetName+"/"+dirName)
        
        if isSplitEnabled:
            #split the data to folds..
            inp = open(datasetName+"/dataset_simplified.csv", "r").readlines()
            random.shuffle(inp)
            numLines = len(inp)
            foldLines = numLines/totalParts
            folds = []
            i = 0
            print "Fold lines: ", foldLines
            for part in range(totalParts):
                newFold = []
                for l in range(foldLines):
                    newFold.append( inp[i] )
                    i += 1
                folds.append(newFold)
            
            #append the remaining rows to the last fold..
            while i < numLines:
                folds[totalParts - 1].append( inp[i] ) 
                i += 1
                
            for i in range(totalParts):
                train = open(datasetName+"/"+dirName+"/complete_train_"+str(i)+".csv", "w")
                test = open(datasetName+"/"+dirName+"/complete_test_"+str(i)+".csv", "w")  
                
                for j in range(totalParts):
                    if i == j:
                        #this is the test set..
                        for row in folds[j]:
                            test.write(row)
                    else:
                        #this is the training set..
                        for row in folds[j]:
                            train.write(row)
                train.close()
                test.close()

        if isPrivSplitEnabled:
            ##############################PART-III##############################
            #create the privilleged datasets..
            for i in range(totalParts):
                print "Splitting the privilleged information!!"
                train = open(datasetName+"/"+dirName+"/complete_train_"+str(i)+".csv", "r").readlines()
                test = open(datasetName+"/"+dirName+"/complete_test_"+str(i)+".csv", "r").readlines()
                
                priv_train = open(datasetName+"/"+dirName+"/priv_train_"+str(i)+".csv", "w")
                pruned_train = open(datasetName+"/"+dirName+"/pruned_train_"+str(i)+".csv", "w")

                priv_test = open(datasetName+"/"+dirName+"/priv_test_"+str(i)+".csv", "w")
                pruned_test = open(datasetName+"/"+dirName+"/pruned_test_"+str(i)+".csv", "w")

                privColumns = privilegedColumns[datasetName]
                split(train, pruned_train, priv_train, privColumns)
                split(test, pruned_test, priv_test, privColumns)

