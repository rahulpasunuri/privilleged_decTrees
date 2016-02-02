# QP solution of SVM+
# based on Fast Optimization Algorithms for Solving SVM+ (D. Pechyony and V. Vapnik)
# Questions regarding this code: N.Quadrianto@sussex.ac.uk

import cvxopt
import random
from cvxopt import matrix
#from cvxopt.blas import dot
from cvxopt.solvers import qp
from vector import CGaussKernel,CLinearKernel,CRBFKernel
import numpy as np

import sys
import os

currPath = os.getcwd()
baseName = os.path.basename(currPath)

sys.path.insert(0, currPath[:-len(baseName)])
from globalConstants import *

#import pdb
def svmplusQP(X,Y,Xstar,C,Cstar):
    n = X.shape[0]
    Y.shape = n,1
    # Compute kernel matrices
    #dk = CRBFKernel();
    #dkstar = CRBFKernel();
    #dK = dk.Dot(X, X)
    #dKstar = dkstar.Dot(Xstar, Xstar)
    #omega_K = 1.0 / np.median(dK.flatten())
    #omega_Kstar = 1.0 / np.median(dKstar.flatten())	
    kernel_K = CLinearKernel() #CGaussKernel(omega_K)
    kernel_Kstar = CLinearKernel() #CGaussKernel(omega_Kstar)
    K = kernel_K.Dot(X,X) 
    Kstar = kernel_Kstar.Dot(Xstar,Xstar)
    #pdb.set_trace()
   
    P = matrix(0.0,(2*n,2*n))
    P[0:n,0:n] = K*np.dot(Y,Y.T)
    P[n:2*n,n:2*n] = Kstar/Cstar
    P = P + np.eye(2*n)*1.e-10
    Q = matrix(0.0, (2*n,1))
    Q[0:n] = np.r_[[-1.0]*n]
    A = matrix(0.0, (2,2*n))
    b = matrix(0.0, (2,1))
    A[1,0:n/2] = 1
    A[1,n/2:n] = -1    
    A[0,n:2*n] = np.r_[[1.0]*n]
    G = matrix(0.0, (2*n,2*n))
    G[0:n,0:n] = np.eye(n)
    G[n:2*n,0:n] = np.diag(np.r_[[-1.0]*n])
    G[n:2*n,n:2*n] = np.eye(n)
    h = matrix(0.0, (2*n,1))
    h[n:2*n] = np.r_[[C]*n]

    #cvxopt.solvers.options['abstol'] = 1e-20	# <<<
    #cvxopt.solvers.options['reltol'] = 1e-20	# <<<
    #cvxopt.solvers.options['feastol'] = 1e-20	# <<<
    print P
    xyz = cvxopt.solvers.qp(matrix(P), matrix(Q), matrix(-1.0*G), matrix(h), matrix(A), matrix(b))

    sol = xyz['x']
    sol = np.array(sol)
    alphas = sol[0:n]
    betahats = sol[n:2*n]
    #betas = betahats - alphas + C
    #alphas[ np.abs(alphas) < 1e-5 ] = 0
    #betas[ np.abs(betas) < 1e-5 ] = 0
    # We compute the bias as explained in Fast Optimization Algorithms for Solving SVM+ (Section 1.5.1)
    Fi = np.dot(K,Y*alphas)
    fi = np.dot(Kstar,betahats)
   # rangeobs = range(n)

    sel_pos = ((alphas.flatten() > 0) & (Y.flatten()==1))
    sel_neg = ((alphas.flatten() > 0) & (Y.flatten()==-1))
    if (sel_pos.shape[0] > 0):
        s_pos = np.mean((1 - fi / Cstar - Fi)[ sel_pos ])
        #print (1 - fi / Cstar - Fi)[ sel_pos ].size
        #print (1 - fi / Cstar - Fi).size
        #print sel_pos.size
    else:
        s_pos = 0
        
    if (sel_neg.shape[0] > 0):
        #TODO: below line is causing the "RuntimeWarning: Mean of empty slice.": FIX IT!!
        #Explanation: Because sel_neg only has false values in it, and the resulting array is empty..
        s_neg = np.mean((-1 + fi / Cstar - Fi)[ sel_neg ])        
        bias = (s_pos + s_neg) / 2.
    else:
        s_neg = 0
        bias = (s_pos + s_neg) / 2.
   
    #print "Printing Result!!"
    #print alphas*Y,bias, "\n"
    return alphas*Y,bias

def svmplusQP_Predict(X,Xtest,alphas,bias):
    # Compute kernel matrices
    #dk = CRBFKernel();
    #dK = dk.Dot(X, Xtest)
    #omega_K = 1.0 / np.median(dK.flatten())
    
    kernel_K = CLinearKernel() #CGaussKernel(omega_K)
    
    K = kernel_K.Dot(X,Xtest) 
    predicted = np.dot(K.T,alphas)+bias
    return predicted
    #return np.sign(predicted)

def convertClassLabel(cl):
    if cl == 0:
        return float(-1)
    else:
        return float(1)

def getRandomPadding():
    return random.uniform(0, 10)

if __name__ == "__main__":

    C = 0.1
    Cstar = 0.1
    
    accuracy = {}
    for dataset in datasets:
        if dataset != "galaxy":
            continue #TODO:remove this!!
        accuracy[dataset] = []    
        print "*"*40
        print "Running Experiments on dataset: ",dataset
        print "*"*40
        for split_num in range(splitCount):
            partAccuracy = 0
            for part in range(totalParts):
                trainFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_train_"+str(part)+".csv", "r").readlines()
                testFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_test_"+str(part)+".csv", "r").readlines()
                privFile = open("../"+dataset+"/"+getSplitName(split_num)+"/priv_train_"+str(part)+".csv", "r").readlines()
                
                X = []
                Y = []
                Xtest = []
                Xstar = []
                Ytest = []
                for row in trainFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    X.append(np.array(row[:-1]))
                    
                    #changing the class labels to -1 and 1
                    Y.append(convertClassLabel(row[len(row) - 1]))

                for row in privFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    Xstar.append(np.array(row[:-1]))
                
                for row in testFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    Xtest.append(np.array(row[:-1]))
                    Ytest.append(convertClassLabel(row[len(row) - 1]))
                    
                X = np.array(X)
                Y = np.array(Y)
                Xstar = np.array(Xstar)
                Xtest = np.array(Xtest)
                                
                duals,bias = svmplusQP(X,Y,Xstar,C,Cstar)
                predicted = svmplusQP_Predict(X,Xtest,duals,bias)
                
                numCorrectPredictions = 0
                
                for rowIndex in range(len(Ytest)):
                    if predicted[rowIndex]*Ytest[rowIndex] > 0:
                        numCorrectPredictions+=1
                                
                #print "Accuracy is: ", (numCorrectPredictions*1.0)/len(Ytest)
                partAccuracy += (numCorrectPredictions*1.0)/len(Ytest)
                exit()#TODO: remove this!!
                '''
                print "*"*40
                print "Printing Predicted\n"
                print "*"*40
                print Xtest.shape
                print predicted.shape
                '''
            accuracy[dataset].append(partAccuracy/totalParts)
        
        print "Accuracy for dataset: ", dataset,": ", round(numpy.mean(accuracy[dataset]), 4), "+- ", round(numpy.std(accuracy[dataset]), 4)
        
        
#error in breast dataset..
