import sys
import csv
import random
import decTree


##############################PART-1##############################
#replace all the non-zero labels to zero..
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


##############################PART-II##############################
#split the dataset to train and test....
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

#Reusing the method from decTree.py to write to csv file 
#decTree.writeResult(predictionsPlusExpectedValues=trainData,fileName="heart_train")
#decTree.writeResult(predictionsPlusExpectedValues=testData,fileName="heart_test")
train = open("original/heart_train.csv", "w")
test = open("original/heart_test.csv", "w")
for l in trainData:
    train.write(l)
train.close()

for l in testData:
    test.write(l)
test.close()



