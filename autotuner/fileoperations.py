# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


import random
import math
import os
from datautil import *

def deleteFiles(settings):
    if settings.file1 != None:
        os.system("rm {}".format(settings.file1))
        
    if settings.file2 != None:
        os.system("rm {}".format(settings.file2))
        
    if settings.file3 != None:
        os.system("rm {}".format(settings.file3))
        
    if settings.file4 != None:
        os.system("rm {}".format(settings.file4))
        

def parseLine(line, dimensions):
    if len(line) > 3 and line[0] != '#':
        d = [(int(math.ceil(float(i))) if int(math.ceil(float(i))) == float(i) else float(i)) for i in line.rstrip().split(' ')]
        
        if len(d) < len(dimensions) + 1:
            return None
        
        return d
    
    return None
        

def readData(inputData, outputData, fileName, dimensions):
    
    try:
        dataFile = open(fileName)
    except:
        print "ERROR: could not open:", fileName, "exiting."
        exit(-1)
    
    dataFile.seek(0)
    for line in dataFile:
        
        d = parseLine(line, dimensions)
        
        if d != None:
            inputData.append(d[:-1])
            outputData.append([d[-1]])
            
    dataFile.close()
            
def createFile1(settings):
    
    configurations = range(0, settings.nConfigurations)
    random.shuffle(configurations)
    
    
    file1 = open(settings.file1, "w+")
    file1.write("{} {}\n".format(settings.nTrainingSamples, len(settings.parameterRanges)))
        
    for n in range(0, settings.nTrainingSamples):
        
        configuration = getConfigurationForNumber(configurations[n], settings.parameterRanges)
        
        parametersString = configurationToString(configuration)
        
        file1.write(parametersString)
        
    file1.close()
    

def createFile3(secondStage, settings):
    
    file3 = open(settings.file3, "w+")
    
    file3.write("{} {}\n".format(len(secondStage), len(secondStage[0])))
    
    for i in range(0, len(secondStage)):
        s = configurationToString(secondStage[i])
        file3.write(s)
        
    file3.close()
    
    
def findDimensions(dataFile):
    
    line = dataFile.readline()
    while line[0] == '#' or len(line) <= 1:
        line = dataFile.readline()
    
    nFields = len(line.split(' '))-1
    values = []
    
    for i in range(0,nFields):
        values.append(set())
    
    dataFile.seek(0)
    for line in dataFile:
        if(len(line) > 3 and line[0] != '#'):
            fields = line.split(' ')
            for i in range(0, len(values)):
                values[i].add(int(fields[i]))
            
            
    return [len(i) for i in values]