# QP solution of SVM+
# based on Fast Optimization Algorithms for Solving SVM+ (D. Pechyony and V. Vapnik)
# Questions regarding this code: N.Quadrianto@sussex.ac.uk

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
    sol = qp(matrix(P), matrix(Q), matrix(-1.0*G), matrix(h), matrix(A), matrix(b))['x']
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
    else:
        s_pos = 0
        
    if (sel_neg.shape[0] > 0):
        s_neg = np.mean((-1 + fi / Cstar - Fi)[ sel_neg ])
        bias = (s_pos + s_neg) / 2.
    else:
        s_neg = 0
        bias = (s_pos + s_neg) / 2.
    
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


    
if __name__ == "__main__":

    C = 0.1
    Cstar = 0.1
    
    #'''
    X = np.random.randn(10,9)
    Xtest = np.random.randn(5,9)
    Xstar = np.random.randn(10,3)
    Y = np.r_[[1]*5, [-1]*5]
    
    #'''
    #exit()
    for dataset in datasets:
        for split_num in range(splitCount):
            for part in range(totalParts):
                trainFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_train_"+str(part)+".csv", "r").readlines()
                testFile = open("../"+dataset+"/"+getSplitName(split_num)+"/pruned_test_"+str(part)+".csv", "r").readlines()
                privFile = open("../"+dataset+"/"+getSplitName(split_num)+"/priv_train_"+str(part)+".csv", "r").readlines()
                
                X = []
                Y = []
                Xtest = []
                Xstar = []
                for row in trainFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    X.append(np.array(row[:-1]))
                    Y.append(row[len(row) - 1])
                
                for row in privFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    Xstar.append(np.array(row[:-1]))
                
                for row in testFile:
                    row = row.split(",")
                    row = [ float(r) for r in row]
                    Xtest.append(np.array(row[:-1]))
                
                X = np.array(X)
                Y = np.array(Y)
                Xstar = np.array(Xstar)
                Xtest = np.array(Xtest)
                
                duals,bias = svmplusQP(X,Y,Xstar,C,Cstar)
                predicted = svmplusQP_Predict(X,Xtest,duals,bias)
                
                print predicted
#    pdb.set_trace() 
