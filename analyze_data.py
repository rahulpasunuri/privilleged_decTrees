from globalConstants import *#file containing all the global constants..


datasetName = "credit"
f = open(datasetName+"/dataset_simplified.csv", "r")
lines = f.readlines()
f.close()

print "Total Number of rows: ", len(lines)
valuesList = {}

for col in nominalColumns[datasetName]:
    valuesList[col] = []
    
    
for l in lines:
    words = l.split(',')
    classIndex = len(words) - 1
    
    for col in range(classIndex):
        if col in nominalColumns[datasetName]:
            if words[col] not in valuesList[col]:
                valuesList[col].append(words[col])
                
for col in valuesList:
    print "Nominal Values for column: ", col," are: ", valuesList[col] 


