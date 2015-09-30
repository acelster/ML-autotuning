# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


import math
import sys
import argparse
import os
import random

from kfold_ann import KFoldAnn
from autotuner import *
from settings import Settings 
from datautil import *
from fileoperations import *

argParser = argparse.ArgumentParser(description = "Machine learning based auto tuner.")
argParser.add_argument('settings_file', nargs='?', default="settings.txt", help="Settings file")
args = argParser.parse_args()

settings = Settings(args.settings_file)

if settings.file1 != None:
    print "Generating", settings.file1, "..."
    createFile1(settings)


if settings.command1 != None:
    print "Executing", settings.command1, "..."
    os.system(settings.command1)
    

print "Reading", settings.file2, "..."

inputData = []
outputData = []
readData(inputData, outputData, settings.file2, settings.parameterRanges)


print "Filtering data..."
filteredInputData = []
filteredOutputData = []

filterData(inputData, outputData, filteredInputData, filteredOutputData)
filteredInputData = transformInput(filteredInputData, settings.parameterRanges)

if settings.nTrainingSamples > len(filteredInputData) or settings.nTrainingSamples > len(inputData):
    print "WARNING: {} training samples requested, but only {} samples in {} and only {} samples after filtering".format(settings.nTrainingSamples, len(inputData), settings.file2, len(filteredInputData))
    print "Reducing number of training samples to {}".format(len(filteredInputData))
    settings.nTrainingSamples = len(filteredInputData)
        

print "Training model and predicting..."
secondStageTimeThreshold, secondStage = tune(filteredInputData, filteredOutputData, settings, KFoldAnn(settings))

print "Generating", settings.file3, "..."
createFile3(secondStage, secondStageTimeThreshold, settings)

print "Executing", settings.command2, "..."
os.system(settings.command2)

print "Reading", settings.file4, "..."
finalConfigs = []
finalTimes = []
readData(finalConfigs, finalTimes, settings.file4, settings.parameterRanges)
validFinalConfigs = []
validFinalTimes = []
filterData(finalConfigs, finalTimes, validFinalConfigs, validFinalTimes)

if settings.keepFiles == 0:
    print "Deleting temporary files..."
    deleteFiles(settings)

index = validFinalTimes.index(min(validFinalTimes))
print
print "Best configuration:"
print configurationToString(validFinalConfigs[index])


