# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


loadError = 0
try:
    from fann2 import libfann
except:
    loadError += 1

if loadError == 1:
    try:
        from pyfann import libfann
    except:
        loadError += 1

if loadError == 2:
    print "ERROR: could not load python fann bindings"
    exit(-1)

import random
import math
import sys
from settings import Settings

class KFoldAnn:
    
    def __init__(self, settings):
        self.k = settings.k 
        self.networks = []
        self.mses = [0]*settings.k
        
        for i in range(0,settings.k):
            self.networks.append(libfann.neural_net())
            self.networks[i].create_standard_array((len(settings.parameterRanges),settings.networkSize,1))
            self.networks[i].set_activation_function_output(libfann.LINEAR)
            
            
    def trainSingleNetwork(self, networkIndex, trainingInput, trainingOutput, testInput, testOutput):
        trainingData = libfann.training_data()
        
        trainingData.set_train_data(trainingInput, trainingOutput)
        
        testData = libfann.training_data()
        testData.set_train_data(testInput, testOutput)
        
        mses = []
        for i in range(0,300):
            for j in range(0,10):
                self.networks[networkIndex].train_epoch(trainingData)
            
            mses.append(self.networks[networkIndex].test_data(testData))
            
            if len(mses) > 10:
                del mses[0]
                
                neg = 0
                for j in range(0,9):
                    if mses[j+1] - mses[j] < 0:
                        neg += 1
                
                if neg < 2:
                    break
                
        self.mses[networkIndex] = mses[9]
                
            
    def train(self, trainingInput, trainingOutput):
        
        inputs = []
        outputs = []
        testInputs = []
        testOutputs = []
        n = len(trainingInput)/self.k
        for i in range(0, self.k):
            inputs.append(trainingInput[0:i*n] + trainingInput[(i+1)*n:])
            testInputs.append(trainingInput[i*n:(i+1)*n])
            outputs.append(trainingOutput[0:i*n] + trainingOutput[(i+1)*n:])
            testOutputs.append(trainingOutput[i*n:(i+1)*n])
            
            
        
        sys.stdout.write("\rTraining: %f%%" % 0.0)
        sys.stdout.flush()
        for i in range(0, self.k):
            self.trainSingleNetwork(i, inputs[i], outputs[i], testInputs[i], testOutputs[i])
            pct = 100*float(i+1)/self.k 
            sys.stdout.write("\rTraining: %f%%" % pct)
            sys.stdout.flush()
        
        print
            
    def runAll(self, inputData):
        predictions = []
        
        for i in range(0, self.k):
            predictions.append([self.networks[i].run(x) for x in inputData])
            
        meanPredictions = []
        for i in range(0, len(inputData)):
            temp = []
            for j in range(0, self.k):
                temp.append(predictions[j][i][0])
                
            meanPredictions.append(self.mean(temp))
            
        return meanPredictions
    
    
    def weightedMean(self, temp):
        w = [1/x for x in self.mses]
        s = sum(w)
        w = [x/s for x in w]
        t = []
        for i in range(0, len(temp)):
            t.append(temp[i]*w[i])
        return sum(t)
    
    
    def mean(self, temp):
        return sum(temp)/len(temp)


    def getErrorEstimate(self):
        est = [math.exp(math.sqrt(x)) - 1 for x in self.mses]
        estErr = np.mean([x*0.5 for x in est])
        
        return estErr
        
            
    def validate(self, validationInput, validationOutput):
        predictions = []
        
        for i in range(0, self.k):
            predictions.append([self.networks[i].run(x) for x in validationInput])
            
        meanPredictions = []
        for i in range(0, len(validationInput)):
            temp = []
            for j in range(0, self.k):
                temp.append(predictions[j][i][0])
                
            meanPredictions.append(self.mean(temp))
            
        
        relativeErrorMean = []
        
        for i in range(0, len(validationOutput)):
            relativeErrorMean.append(abs((meanPredictions[i] - validationOutput[i][0])/validationOutput[i][0]))
                
               
        unscaledPredictions = []
        unscaledValidation = []
        for i in range(0, len(validationOutput)):
            unscaledPredictions.append(math.exp(meanPredictions[i]))
            unscaledValidation.append(math.exp(validationOutput[i][0]))
            
        unscaledRelativeError = []
        for i in range(0, len(unscaledPredictions)):
            unscaledRelativeError.append(abs((unscaledPredictions[i] - unscaledValidation[i])/unscaledValidation[i]))
                
        return self.mean(unscaledRelativeError)
        
                
            
                           
                           
