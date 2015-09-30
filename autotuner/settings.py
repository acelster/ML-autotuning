# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


class Settings:
    
    def __init__(self, settingsFileName=None):
        self.setDefaults()
        
        if settingsFileName == None:
            return
        
        try:
            settingsFile = open(settingsFileName)
        except:
            print "Could not open", settingsFileName, "exiting..."
            exit(-1)
        
        for line in settingsFile:
            if line[0] == '#':
                continue
            
            l = line.rstrip().split(":")
            
            if l[0] == "PARAMETER_RANGES":
                self.parameterRanges = [int(p) for p in l[1].split(' ')]
            if l[0] == "COMMAND1":
                self.command1 = l[1]
            if l[0] == "COMMAND2":
                self.command2 = l[1]
            if l[0] == "FILE1":
                self.file1 = l[1]
            if l[0] == "FILE2":
                self.file2 = l[1]
            if l[0] == "FILE3":
                self.file3 = l[1]
            if l[0] == "FILE4":
                self.file4 = l[1]
            if l[0] == "NETWORK_SIZE":
                self.networkSize = int(l[1])
            if l[0] == "N_TRAINING_SAMPLES":
                self.nTrainingSamples = int(l[1])
            if l[0] == "N_SECOND_STAGE_MIN":
                self.nSecondStageMin = int(l[1])
            if l[0] == "N_SECOND_STAGE":
                self.nSecondStage = int(l[1])
                self.useSecondStageAbs = True
            if l[0] == "N_SECOND_STAGE_MAX":
                self.nSecondStageMax = int(l[1])
            if l[0] == "SECOND_STAGE_THRESHOLD":
                self.secondStageThreshold = float(l[1])
                self.useSecondStageThreshold = True
            if l[0] == "KEEP_FILES":
                self.keepFiles = int(l[1])
            if l[0] == "K":
                self.k = int(l[1])
                
        self.computeNConfigurations()
        
        self.checkSettings()
        
        
    def computeNConfigurations(self):
        n = 1
        for i in self.parameterRanges:
            n *= i
            
        self.nConfigurations = n
        
        
    def checkSettings(self):
        if self.networkSize <= 0:
            print "ERROR: network size must be positive"
            exit(-1)
        
        if self.nTrainingSamples <= 0:
            print "ERROR: number of training samples must be positive"
            exit(-1)
            
        if self.nSecondStage <= 0:
            print "ERROR: number of samples in second stage must be positive"
            exit(-1)

        if self.nSecondStageMin <= 0:
            print "ERROR: number of samples in second stage must be positive"
            exit(-1)

        if self.nSecondStageMax <=0:
            print "ERROR: number of samples in second stage must be positive"
            exit(-1)

        if self.nSecondStageMax <= self.nSecondStageMin:
            print "ERROR: maximum number of samples in second stage must be larger than minimum"
            exit(-1)

        if self.secondStageThreshold < 0.0 or self.secondStageThreshold > 1.0:
            print "ERROR: second stage threshold must be between 0 and 1.0"
            exit(-1)
            
        if self.useSecondStageAbs and self.useSecondStageThreshold:
            print "WARNING: Both fixed and threshold based second stage size specified. Defaulting to fixed." 

        if not (self.useSecondStageAbs or self.useSecondStageThreshold):
            print "WARNING: Neither fixed and threshold based second stage size specified. Defaulting to fixed." 
            
        if self.k <= 0:
            print "ERROR: k must be positive"
            exit(-1)
            
        if self.file2 == None:
            print "ERROR: must specify file 2."
            exit(-1)
            
        if self.file4 != None and self.command2 == None:
            print "WARNING: file4 specified, but command2 not specified. file4 will be ignored."
            self.file4 = None
            
        if len(self.parameterRanges) == 0:
            print "ERROR: number of parameters cannot be 0"
            exit(-1)
            
        if self.nConfigurations == 0:
            print "ERROR: parameter ranges cannot be 0"
            exit(-1)
            
        if self.nConfigurations < self.nTrainingSamples:
            print "WARNING: {} training samples requested, but only {} unique exists.".format(self.nTrainingSamples, self.nConfigurations)
            print "Reducing numer of training samples to {}".format(self.nConfigurations)
            self.nTrainingSamples = self.nConfigurations

            
    def setDefaults(self):
        self.parameterRanges = []
        self.nConfigurations = 0
        self.command1 = None
        self.command2 = None
        self.file1 = None
        self.file2 = None
        self.file3 = None
        self.file4 = None
        self.networkSize = 30
        self.nTrainingSamples = 100
        self.nSecondStage = 10
        self.nSecondStageMax = 10
        self.nSecondStageMin = 100
        self.secondStageThreshold = 0.1
        self.keepFiles = 0
        self.k = 10
        
        self.useSecondStageAbs = False
        self.useSecondStageThreshold = False
        
                
    def printSettings(self):
        print "PARAMETER_RANGES", self.parameterRanges
        print "N_CONFIGURATIONS", self.nConfigurations
        print "COMMAND1", self.command1
        print "COMMAND2", self.command2
        print "FILE1", self.file1
        print "FILE2", self.file2
        print "FILE3", self.file3
        print "FILE4", self.file4
        print "NETWORK_SIZE", self.networkSize
        print "N_TRAINING_SAMPLES", self.nTrainingSamples
        print "N_SECOND_STAGE", self.nSecondStage
        print "KEEP_FILES", self.keepFiles
        print "K", self.k
            
