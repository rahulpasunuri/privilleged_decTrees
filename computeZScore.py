import math
#import scipy
vals = []

dtCompare = "DT vs DT+"
dtPlusCompare = "GB vs. GB+"
#formal old_mean, old_std, new_mean, new_std, testSetSize, datasetName, measure


vals.append([70.96, 0.76, 72.58, 0.34, 59, "Heart Disease", dtCompare])
vals.append([77.76,1.23, 79.78, 00.89, 59, "Heart Disease", dtPlusCompare])

vals.append([81.53, 1.86, 83.70, 1.37, 42, "Glass Identification", dtCompare])
vals.append([80.16, 0.97, 83.48, 5.6, 42, "Glass Identification", dtPlusCompare])

vals.append([77.38, 0.9, 77.45, 0.81, 345, "Car Evaluation", dtCompare])
vals.append([70.03, 0, 72.49, 0.51, 345, "Car Evaluation", dtPlusCompare])

vals.append([91.31, 1.08, 92.14, 1.40, 67, "Ecoli", dtCompare])
vals.append([92.62, 0.39, 93.27, 0.3, 67, "Ecoli", dtPlusCompare])

vals.append([79.8, 1.16, 81.2, 1.6, 20, "Fertility", dtCompare])
vals.append([79.6, 1.85, 80.4, 1.15, 20, "Fertility", dtPlusCompare])

vals.append([68.67, 1.04, 70.36, 1.04, 153, "Pima Indians Diabetes", dtCompare])
vals.append([71.12, 0.67, 73.65, 0.79, 153, "Pima Indians Diabetes", dtPlusCompare])

vals.append([89.65, 0.85, 89.75, 0.88, 20, "Seeds", dtCompare])
vals.append([89.34, 0.87, 91.17, 1.13, 20, "Seeds", dtPlusCompare])

vals.append([66.53, 1.17, 68.4, 1.81, 39, "Galaxy", dtCompare])
vals.append([72, 0.81, 71.8, 0.39, 39, "Galaxy", dtPlusCompare])

vals.append([75.28, 2.45, 75.14, 3.04, 28, "Hepatitis", dtCompare])
vals.append([79, 1.04, 82.58 , 5.24,28, "Hepatitis", dtPlusCompare])

vals.append([84.65, 2.38, 83.81, 3.8, 38, "Flags", dtCompare])
vals.append([83.65, 0.96, 86.67 , 1.17,38, "Flags", dtPlusCompare])

vals.append([93.07, 0.9, 92.93, 0.9, 30, "Iris", dtCompare])
vals.append([93.07, 00.53, 94.67, 1.63,30, "Iris", dtPlusCompare])

vals.append([79.78, 0.89, 80, 1, -1, "Heart", "GB+ vs. SVM+"])

vals.append([75.14, 3.04, 79, 1.04, -1, "Hepatitis", "DT+ vs. GB"])

vals.append([82.58, 5.24, 79.66, 0.19, -1, "Hepatitis", "GB+ vs. SVM+"])

def computeScore(p):
    if math.sqrt(p[1]*p[1]+p[3]*p[3]) == 0:
        print "Error Error Error Error Error: No STD in both the distributions"
        return -100 
    zScore = (p[2] - p[0])/math.sqrt(p[1]*p[1]+p[3]*p[3])
    #zScore = zScore*math.sqrt(p[4])
    zScore = zScore*math.sqrt(5)
    return zScore 
'''
vals.append([10, 1, 10, 1, 100, "heart", dtCompare])
vals.append([10, 1, 10, 1, 100, "heart", dtPlusCompare])
'''
def main():
    for p in vals:
        zScore = computeScore(p)
        if len(str(zScore)) > 4:
            zScore = round(zScore, 2)
        print "Z-score in dataset "+p[5]+" of "+str(p[4])+" examples with measure - "+p[6]+" is: ",zScore
        print
        
#Execution begins here
if __name__ == "__main__" : main()
