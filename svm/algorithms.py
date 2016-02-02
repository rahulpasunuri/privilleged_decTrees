from sklearn import svm

class SVM:
    def __init__(self):
        self.model=None
        self.prediction=None

    def learn(self,Xtrain,Ytrain):
        #model=svm.SVR(kernel='sigmoid',gamma='auto',C=50)
        #print('\n Kernel : {0}').format(self.kernelType)
        #model=svm.SVC(kernel='linear',C=self.C,class_weight={1.0:1.0,0.0:2.0})
        model=svm.SVC(kernel='linear')
        model.fit(Xtrain,Ytrain)
        self.model=model
        print('\n Model : {0}').format(self.model)

    def predict(self,Xtest):
        model=self.model
        self.prediction=model.predict(Xtest)
        return self.prediction

