# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


import math

def transformInput(inputData, dimensions):
    
    newInputData = []
    for i in range(0, len(inputData)):
        
        r = [0]*len(dimensions)
        for j in range(0, len(dimensions)):
            if dimensions[j] == 2:
                #print j, i
                r[j] = 1 if inputData[i][j] == 1 else -1
            else:
                r[j] = inputData[i][j] - dimensions[j]/2
        
        newInputData.append(r)
        
    return newInputData

def untransformSingle(entry, dimensions):
    
    new = [0]*len(dimensions)
    for i in range(0, len(dimensions)):
        if dimensions[i] == 2:
            new[i] = 1 if entry[i] == 1 else 0
        else:
            new[i] = entry[i] + dimensions[i]/2
            
    return new

def filterData(inputData, outputData, filteredInputData, filteredOutputData):
    
    for i in range(0, len(outputData)):
        if outputData[i][0] > 0:
            if outputData[i][0] <= 1000000000:
                filteredInputData.append(inputData[i])
                filteredOutputData.append(outputData[i])
                

def getConfigurationForNumber(number, parameterRanges):
    nParameters = len(parameterRanges)
    cumulutative = [0] * nParameters
    parameters = [0] * nParameters
    cumulutative[nParameters-1] = 1
    
    for i in range(nParameters-2, -1, -1):
        cumulutative[i] = cumulutative[i+1]*parameterRanges[i+1]
        
    for i in range(0, nParameters):
        parameters[i] = number/cumulutative[i]
        number = number - (parameters[i] * cumulutative[i])

    return parameters

def configurationToString(configuration):
    
    s = ""
    for i in configuration:
        s += (str(i) + " ")
    s += "\n"
    
    return s
