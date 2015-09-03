# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


import math
import random
import copy
import sys
from kfold_ann import KFoldAnn
from datautil import * 


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
    
    secondStageConfigs = []
    
    print "Finding best predictions..."
    for i in range(0, settings.nSecondStage):
        index = predictions.index(min(predictions))
        combo = untransformSingle(allInputCombinations[index], settings.parameterRanges)

        secondStageConfigs.append(combo)
        del predictions[index]
        del allInputCombinations[index]
        
    return secondStageConfigs