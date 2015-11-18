def computeStats(predictedLabels):
    
    truePos = {}
    trueNeg = {}
    falsePos = {}
    falseNeg = {}
    
    labels = []
    predLabels = []
    
    for label in predictedLabels:
        actual = float(label[1])
        
        if actual not in labels:
            labels.append(actual)

    for label in labels:
        #initialize the value in all the dicts..
        if label not in truePos:
            truePos[label] = 0
        if label not in trueNeg:
            trueNeg[label] = 0
        if label not in falsePos:
            falsePos[label] = 0
        if label not in falseNeg:
            falseNeg[label] = 0        
        
    for label in predictedLabels:
        actual = float(label[1])
        predicted = float(label[0])
        
        #check false positives and true positives..
        if actual == predicted:
            truePos[actual] += 1
            for curr in labels:
                if curr != actual:
                    trueNeg[curr] += 1 
        else:
            falsePos[predicted] += 1
            falseNeg[actual] += 1    
            
    precision = {}
    recall = {}
    accuracy = {}
    
    for label in labels:
        if truePos[label] + falsePos[label] != 0:
            precision[label] = float(truePos[label])/(truePos[label] + falsePos[label])
        else:
            precision[label] = 0

        accuracy[label] = float(truePos[label] + trueNeg[label])/len(predictedLabels)

        if truePos[label] + falseNeg[label] != 0:
            recall[label] = float(truePos[label]) / (truePos[label] + falseNeg[label])
        else:
            recall[label] = 0

    return precision, recall, accuracy
    
def computeAccuracy(predictedLabels):
    correct = 0
    for row in predictedLabels:
        if float(row[0]) == float(row[1]):
            correct+=1
    return float(correct)/len(predictedLabels)

