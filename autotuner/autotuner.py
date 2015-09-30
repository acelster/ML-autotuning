# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


import math
import random
import copy
import sys
import scipy as sp
import scipy.stats
from kfold_ann import KFoldAnn
from datautil import * 

def probIsMin(mu_a, mu_b, stdev_a, stdev_b):
    #The probability that a is less than b, e.g. a < b, e.g. b - a > 0
    
    #mu =  mu_b-mu_a
    #var = stdev_a**2 + stdev_b**2 
    #return 1.0 - sp.stats.norm(loc=mu, scale=math.sqrt(var)).cdf(0)

    mu =  mu_b
    var = stdev_a**2 + stdev_b**2 
    return 1.0 - sp.stats.norm(loc=mu, scale=math.sqrt(var)).cdf(0 + mu_a)

def invProbIsMin(mu_b, stdev_a, stdev_b, threshold):
    mu =  mu_b
    var = stdev_a**2 + stdev_b**2 

    return sp.stats.norm(loc=mu, scale=math.sqrt(var)).ppf(1-threshold)

def getSecondStageTimeThreshold(bestEstimate, kfa, settings):
    stdevBest = kfa.getErrorEstimate()*bestEstimate
    stdevUpper = kfa.getErrorEstimate()*bestEstimate #THIS IS WRONG, should be the the time we're finding...
    return invProbIsMin(bestEstimate, stdevUpper, stdevBest, settings.secondStageThreshold)



def getSubset(full1, full2, subset1, subset2, subsetSize):
    for i in range(0, subsetSize):
        index = int(math.floor(random.random() * len(full1)))
        subset1.append(full1.pop(index))
        subset2.append(full2.pop(index))

            
def getAllInputCombinations(settings):
    
    return [getConfigurationForNumber(x, settings.parameterRanges) for x in range(0,settings.nConfigurations)]
    


def tune(inputData, outputData, settings, kfa):
    
    outputData = [[math.log(x[0])] for x in outputData]
    
    ti = []
    to = []
    
    getSubset(inputData, outputData, ti, to, settings.nTrainingSamples)
    
    kfa.train(ti, to)
    
    print "Predicting..."
    allInputCombinations = getAllInputCombinations(settings)
    allInputCombinations = transformInput(allInputCombinations, settings.parameterRanges)
    
    predictions = kfa.runAll(allInputCombinations)
    
    bestPredictedTime = min(predictions)
    secondStageTimeThreshold = getSecondStageTimeThreshold(math.exp(bestPredictedTime), kfa, settings)
    
    print "Finding best predictions..."
    
    predictions, allInputCombinations = [list(t) for t in zip(*sorted(zip(predictions,allInputCombinations)))]
    secondStageConfigs = [untransformSingle(x, settings.parameterRanges) for x in allInputCombinations]
    
        
    return secondStageTimeThreshold, secondStageConfigs
