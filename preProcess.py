import sys
import csv
import random
import decTree
import os
from globalConstants import * #file containing all the global constants..
import sklearn
from sklearn.preprocessing import *
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

missingFeatures = {}
nominalValues = {}
for datasetName in datasets:
    if isSimplificationEnabled:
        ##############################PART-1##############################
        #replace all the non-zero labels to zero..
        print "Simplifying the dataset"
        f = open(datasetName+"/dataset.csv", "r")
        out = open(datasetName+"/dataset_simplified.csv", "w")
        lines = f.readlines()
        lineNum = 0
        countMissing = 0
        for l in lines:
            lineNum += 1
            l = l.strip()
            if l == "":
                continue # skip empty lines..
            if datasetName == "heart" or datasetName == "heart1" or datasetName == "heart2" or datasetName == "heart3" or datasetName == "heart4":
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
                    #words[classIndex] = '2'
                    words[classIndex] = '0'

            elif "glass_binary" in datasetName:
                words = l.strip().split(",")
                #ignore the first columns, which is just the id of the rows..
                words = words[1:]
                classIndex = len(words) - 1
                
                #building glasses vs. non building glasses.
                if words[classIndex] == "0" or words[classIndex] == "1":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'
            elif datasetName == "car":
                #data cleaning..replacing the below string to make the feature numeric..
                l = l.replace("5more", "5")
                l = l.replace("more", "5") # replcaing "more" string with the number 5 
                words = l.strip().split(",")
                classIndex = len(words) - 1
                #convert the class index..
                if words[classIndex] == "unacc":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'

            elif datasetName == "car_continuous":
                #data cleaning..replacing the below string to make the feature numeric..
                l = l.replace("5more", "5")
                l = l.replace("more", "5") # replcaing "more" string with the number 5 
                
                l = l.replace("vhigh", "13") 
                l = l.replace("high", "12") 
                l = l.replace("big", "11") 
                l = l.replace("med", "10") 
                l = l.replace("small", "10") 
                l = l.replace("low", "9")
                
                words = l.strip().split(",")
                classIndex = len(words) - 1
                
                #convert the class index..
                if words[classIndex] == "unacc":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'
                
            elif datasetName == "census":
                words = l.strip().split(",")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1
                #convert the class index..
                if words[classIndex] == "<=50K":
                    words[classIndex] = '0'
                elif words[classIndex] == ">50K":
                    words[classIndex] = '1'

            elif datasetName == "credit":
                if "?" in l:
                    #skip rows which have missing values..
                    countMissing += 1
                    continue
                words = l.strip().split(",")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1

                #convert the class index..
                if words[classIndex] == "+":
                    words[classIndex] = '1'
                elif words[classIndex] == "-":
                    words[classIndex] = '0'
                else:
                    print lineNum
                    print words
                    print "ERROR!!!!  "*4
                    exit()
            elif datasetName == "ecoli_binary":
                words = []
                words_extra = l.strip().split(" ")
                #ignore column 1, because it is too specific..
                words_extra = words_extra[1:]
                for wor in words_extra:
                    wor = wor.strip()
                    if wor != "":
                        words.append(wor)
                #different set of classes = [ "cp", "im", "imS", "imL", "imU", "om", "omL", "pp"]
                #"cp" has the highest number of class occurrences, so lets do cp vs. non cp..
                classIndex = len(words) - 1
                if words[classIndex] == "cp":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'

            elif datasetName == "hepatitis":

                words = l.strip().split(",")
                words = [ wor.strip() for wor in words]
                
                #take the class from first to last column of the row..
                cl = words[0]
                words = words[1:]
                words.append(cl)
                
                classIndex = len(words) - 1                
                for i in range(classIndex):
                    if words[i] == "?":
                        if i not in missingFeatures:
                            missingFeatures[i] = 1
                        else:
                            missingFeatures[i] = missingFeatures[i] + 1
                     
                #ignore features with missing values here!!!
                currMissingFeatures = [7, 8, 14, 16, 17]
                newWords = []
                for i in range(len(words)):
                    if i not in currMissingFeatures:
                        newWords.append(words[i])
                tempMissingString = ",".join(newWords)
                words = newWords
                classIndex = len(words) - 1  
                if "?" in tempMissingString:
                    countMissing += 1     
                    continue

                #TODO: fix this..
                #convert the class index..
                if words[classIndex] == "1":
                    words[classIndex] = '1'
                elif words[classIndex] == "2":
                    words[classIndex] = '0'
                else:
                    print lineNum
                    print words
                    print "ERROR!!!!  "*4
                    exit()
            elif datasetName == "galaxy":
                words = l.strip().split(",")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1
                #convert the class index..
                if words[classIndex] == "yes":
                    words[classIndex] = '1'
                elif words[classIndex] == "no":
                    words[classIndex] = '0'
                else:
                    print lineNum
                    print words
                    print "ERROR!!!!  "*4
                    exit()    
            elif datasetName == "fertility":
                words = l.strip().split(",")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1
                #convert the class index..
                if words[classIndex] == "N":
                    words[classIndex] = '1'
                elif words[classIndex] == "O":
                    words[classIndex] = '0'

            elif datasetName == "nursery":
                words = l.strip().replace("more", "4").split(",")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1

                #convert the class index..
                if words[classIndex] == "not_recom" or words[classIndex] == "recommend" or words[classIndex] == "very_recom":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'
                
                for colId in range(len(words)):
                    if colId in nominalColumns[datasetName]:
                        if colId not in nominalValues:
                            nominalValues[colId] = []
                        if words[colId] not in nominalValues[colId]:
                            nominalValues[colId].append(words[colId])

            elif datasetName == "seeds":
                words = l.strip().split("\t")
                words = [ wor.strip() for wor in words]
                classIndex = len(words) - 1
                
                if classIndex != 7:
                    continue
                isSkipLine = False
                for wor in words:
                    if wor.strip() == "":
                        isSkipLine = True
                        break #missing values..ignore this line..
                        
                if isSkipLine:
                    continue

                #print words
                #convert the class index..
                if words[classIndex] == "1" or words[classIndex] == "2":
                    words[classIndex] = '0'
                else:
                    words[classIndex] = '1'

            elif datasetName == "flags":
                words = l.strip().split(",")
                words = [ wor.strip() for wor in words][1:] #ignore the country column (0th column..)
                newList = []
                classIndex = 6
                for col in range(len(words)):
                    if col != classIndex: #col6 is the class index..
                        newList.append(words[col])
                    
                newList.append(words[classIndex])
                words = newList

                
                classIndex = len(words) - 1
                #convert the class index..
                if words[classIndex] == "1" or words[classIndex] == "0":
                    words[classIndex] = '1'
                else:
                    words[classIndex] = '0'                                         
            else:
                words = l.strip().split(",")
                classIndex = len(words) - 1 
                
            out.write(",".join(words)+"\n")
        out.close()
        print "Missing values in the Dataset: "+datasetName+" is: ", countMissing
        print "Missing Features in the Dataset: "+datasetName+"is: ", missingFeatures
    
    for key in nominalValues:
        print key, nominalValues[key]

    if datasetName == "car_continuous":
        #enc = OneHotEncoder(categorical_features=[0, 1, 4, 5])
        enc = OneHotEncoder()
        x = [ r.strip().split(",") for r in open(datasetName+"/dataset_simplified.csv", "r").readlines()]
        data = []
        for row_actual in x:
            nominalRow = []
            for colIndex in nominalColumns["car"]:
                nominalRow.append(row_actual[colIndex])
            data.append(nominalRow)
        #print data
        enc.fit(data)
        data=enc.transform(data)
        #print len(data)
        data = data.toarray().tolist()
        out = open(datasetName+"/dataset_simplified.csv", "w")
        for rowIndex in range(len(data)):
            newRow = data[rowIndex]
            for colIndex in range(len(x[rowIndex])):
                if colIndex not in nominalColumns["car"]:
                    newRow.append(x[rowIndex][colIndex])
            words = [ str(r) for r in newRow]
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

