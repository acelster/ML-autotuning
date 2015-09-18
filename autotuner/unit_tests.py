# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


from settings import Settings
from autotuner import *
from datautil import * 
import fileoperations
import unittest

class TestSettings(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_computeNConfigurations(self):
        settings= Settings()
        settings.parameterRanges = [2,4,5,3,7]
        settings.computeNConfigurations()
        
        self.assertEqual(2*4*5*3*7, settings.nConfigurations)
        

class TestFileoperations(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_parseLine(self):
        dimensions = [4,4,4,4]
        line = "2 3 1 0 34.4\n"
        
        d = fileoperations.parseLine(line, dimensions)
        
        self.assertEqual([2,3,1,0,34.4], d)
        
        
    def test_parseLine_comment_blank(self):
        dimensions = [4,4,4,4]
        line_blank = "\n"
        line_comment = "#2 3 1 0 34.4\n"
        
        d_blank = fileoperations.parseLine(line_blank, dimensions)
        d_comment = fileoperations.parseLine(line_comment, dimensions)
        
        self.assertEqual(None, d_blank)
        self.assertEqual(None, d_comment)
        
        
    def test_parseLine_short(self):
        dimensions = [4,4,4,4]
        line_short = "2 3 1 4\n"
        
        d_short = fileoperations.parseLine(line_short, dimensions)
        
        self.assertEqual(None, d_short)
        

class TestTransformer(unittest.TestCase):

    def setUp(self):
        return None
        
    def test_unstransformSingle(self):
        entry = [-1,1, -3, 2]
        dimensions = [2,2,8,8]
        self.assertEqual([0,1,1,6], untransformSingle(entry, dimensions))
        
        
    def test_transformInput(self):
        inputData = [[1,0,3,0],[0,7,6,1]]
        dimensions = [2,8,8,2]
        newInputData = transformInput(inputData,dimensions)
        
        self.assertEqual([1,-4,-1,-1], newInputData[0])
        self.assertEqual([-1,3,2,1], newInputData[1])
        
        
    def test_filterData(self):
        inputData = [[1],[2],[3],[4]]
        outputData = [[-1],[1],[-1],[1]]
        filteredInputData = []
        filteredOutputData = []
        
        filterData(inputData, outputData, filteredInputData, filteredOutputData)
        
        self.assertEqual(2, len(filteredInputData))
        self.assertEqual(2, len(filteredOutputData))
        self.assertEqual(2, filteredInputData[0][0])
        self.assertEqual(4, filteredInputData[1][0])
        
        
    def test_getConfigurationForNumber(self):
        
        parameterRanges = [4,2,4,3]
        
        self.assertEqual([0,0,0,0], getConfigurationForNumber(0, parameterRanges))
        self.assertEqual([0,1,0,1], getConfigurationForNumber(13, parameterRanges))
        self.assertEqual([2,1,3,0], getConfigurationForNumber(48+12+9, parameterRanges))
        self.assertEqual([3,1,3,2], getConfigurationForNumber(4*2*4*3-1, parameterRanges))
        
        
    def test_configurationToString(self):
        c = [2,4,3,1,6]
        
        s = configurationToString(c)
        
        self.assertEqual("2 4 3 1 6 \n", s)


class  Mock_KFoldAnn:
    
    def train(self, a, b):
        pass
    
    def runAll(self, inputData):
        outputData = [sum(x) for x in inputData]
        
        return outputData
        
class TestAutotuner(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_probIsMin(self):
        
        a = probIsMin(1,2,1,1)
        self.assertAlmostEqual(a,1-0.2398,places=4)
        
        a = probIsMin(0.5,1.5,1,1)
        self.assertAlmostEqual(a, 1-0.2398,places=4)
        
        a = probIsMin(1,2,0.5,0.5)
        self.assertAlmostEqual(a,0.9214,places=4)
        
        a = probIsMin(1,2,0.3,0.5)
        self.assertAlmostEqual(a,0.9568,places=4)
        
        a = probIsMin(2,1,0.5,0.3)
        self.assertAlmostEqual(a,1-0.9568,places=4)

    def test_invProbIsMin(self):
        a = invProbIsMin(2, 1, 1, 1-0.2398)
        self.assertAlmostEqual(a,1.000,places=3)
        
        
    def test_getSubset(self):
        full1 = [[1],[2],[3],[4],[5]]
        full2 = [[-1],[-2],[-3],[-4],[-5]]
        sub1 = []
        sub2 = []
        
        getSubset(full1, full2, sub1, sub2, 3)
        
        self.assertEqual(3,len(sub1))
        self.assertEqual(3,len(sub2))
        self.assertEqual(2,len(full1))
        self.assertEqual(2,len(full2))
        
        for i in range(0,3):
            self.assertTrue(sub1[i][0] == -sub2[i][0])
            
        for i in range(1,6):
            self.assertTrue([i] in sub1 or [i] in full1)
            self.assertTrue([-i] in sub2 or [-i] in full2)
            
    
    def test_getAllInputCombinations_small(self):
        settings= Settings()
        settings.parameterRanges = [2,2]
        settings.computeNConfigurations()
        
        allCombinations = getAllInputCombinations(settings)
        
        self.assertEqual([0,0], allCombinations[0])
        self.assertEqual([0,1], allCombinations[1])
        self.assertEqual([1,0], allCombinations[2])
        self.assertEqual([1,1], allCombinations[3])
        
    
    def test_getAllInputCombinations_big(self):
        settings= Settings()
        settings.parameterRanges = [3,4,2,5]
        settings.computeNConfigurations()
        
        allCombinations = getAllInputCombinations(settings)
        
        self.assertEqual(3*4*2*5, len(allCombinations))
        self.assertEqual([0,0,0,0], allCombinations[0])
        self.assertEqual([2,3,1,4], allCombinations[-1])
        self.assertTrue([1,1,1,1] in allCombinations)
        self.assertFalse([3,0,0,0] in allCombinations)
        self.assertFalse([0,4,0,0] in allCombinations)
        self.assertFalse([0,0,2,0] in allCombinations)
        self.assertFalse([0,0,0,5] in allCombinations)
        
    def test_tune(self):
        inputData = []
        outputData = []
        settings= Settings()
        settings.parameterRanges = [2,3,4]
        settings.computeNConfigurations()
        settings.nTrainingSamples = 0
        settings.nSecondStage = 3
        
        secondStageConfigs = tune(inputData, outputData, settings, Mock_KFoldAnn())
        
        self.assertEqual(3, len(secondStageConfigs))
        self.assertEqual([0,0,0], secondStageConfigs[0])
        self.assertEqual(1, sum(secondStageConfigs[1]))
        self.assertEqual(1, sum(secondStageConfigs[2]))
        
            
if __name__ == '__main__':
	unittest.main()
