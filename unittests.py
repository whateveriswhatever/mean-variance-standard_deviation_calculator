import unittest
import numpy as np
# from meanVarianceStandardDeviationCalculator import Matrix

class MatrixStatisticalCalculation:
    def __init__(self, npArr: np.array):
        # Reshaping the primary NumPy array into 3x3 matrix
        self._input = self.matrixTransformer(npArr)
        
    def matrixTransformer(self, npArr: np.array):
        if npArr.shape != (3, 3) and npArr.shape == (9, ):
            npArr = npArr.reshape(3, 3)
            return npArr
        
        elif npArr.shape == (3, 3):
            return npArr

        else:
            raise ValueError("Input Numpy Array must contain 9 numbers!!!")
    
    def getStd(self) -> list:
        data = []
        
        # Vertical axis
        stdAtCols = [self._input[:, 0].std(), self._input[:, 1].std(), self._input[:, 2].std()]
        print("Std at cols: {}".format(stdAtCols))
        
        # Horizontal axis
        stdAtRows = [self._input[0, :].std(), self._input[1, :].std(), self._input[2, :].std()]
        print("Std at rows: {}".format(stdAtRows))
        
        # Standard Deviation of the entire matrix
        stdMatrix = self._input.std()
        print("Std at matrix: {}".format(stdMatrix))
        
        data.append(stdAtCols)
        data.append(stdAtRows)
        data.append(stdMatrix)
        
        return data
    
    def getMean(self) -> list:
        data = []
        
        # Vertical axis
        meanAtCols = [self._input[:, 0].mean(), self._input[:, 1].mean(), self._input[:, 2].mean()]
        
        # Horizontal axis
        meanAtRows = [self._input[0, :].mean(), self._input[1, :].mean(), self._input[2, :].mean()]
        
        # Mean of the entire matrix
        meanMatrix = self._input.mean()
        
        data.append(meanAtCols)
        data.append(meanAtRows)
        data.append(meanMatrix)
        
        return data

    def getVariance(self) -> list:
        data = []
        
        # Vertical axis
        # varianceAtCols = self._input.var(axis=0)
        varianceAtCols = [self._input[:, 0].var(), self._input[:, 1].var(), self._input[:, 2].var()]
        
        # Horizontal axis
        varianceAtRows = [self._input[0, :].var(), self._input[1, :].var(), self._input[2, :].var()]
        
        # Variance of the entire matrix
        varianceMatrix = self._input.var()
        
        data.append(varianceAtCols)
        data.append(varianceAtRows)
        data.append(varianceMatrix)
        
        return data
    
    def getMin(self) -> list:
        data = []
        
        # Vertical axis
        # minAtCols = self._input.min(axis=0)
        minAtCols = [self._input[:, 0].min(), self._input[:, 1].min(), self._input[:, 2].min()]
        
        
        # Horizontal axis
        # minAtRows = self._input.min(axis=1)
        minAtRows = [self._input[0, :].min(), self._input[1, :].min(), self._input[2, :].min()]
        
        # Minimum value of the entire matrix
        minMatrix = self._input.min()
        
        # print("minAtCols {} -> data type: {}".format(minAtCols, minAtCols.dtype))
        # print("minAtRows {} -> data type: {}".format(minAtRows, minAtRows.dtype))
        # print("minMatrix {} -> data type: {}".format(minMatrix, minMatrix.dtype))
        
        data.append(minAtCols)
        data.append(minAtRows)
        data.append(minMatrix)
        
        # print("data: {}".format(data))
        
        return data
    
    def getMax(self) -> list:
        data = []
        
        # Vertical axis
        maxAtCols = list(self._input.max(axis=0))
        
        # Horizontal axis
        maxAtRows = list(self._input.max(axis=1))
        
        # Maximum value of the entire matrix
        maxMatrix = self._input.max()
        
        data.append(maxAtCols)
        data.append(maxAtRows)
        data.append(maxMatrix)
        
        # print("maxAtCols {} -> data type: {}".format(maxAtCols, maxAtCols.dtype))
        # print("maxAtRows {} -> data type: {}".format(maxAtRows, maxAtRows.dtype))
        # print("maxMatrix {} -> data type: {}".format(maxMatrix, maxMatrix.dtype))
        
        return data
    
    def getSum(self) -> list:
        data = []
        
        # Vertical axis
        sumAtCols = list(self._input.sum(axis=0))
        
        # Horizontal axis
        sumAtRows = list(self._input.sum(axis=1))
        
        # Sum of the entire matrix
        sumMatrix = self._input.sum()
        
        
        data.append(sumAtCols)
        data.append(sumAtRows)
        data.append(sumMatrix)
        
        return data
    
    def getStatisticalInfor(self) -> dict:
        meanData = self.getMean()
        varianceData = self.getVariance()
        stdData = self.getStd()  
        minData = self.getMin()
        maxData = self.getMax()
        sumData = self.getSum()

        calculations = {
            "mean": meanData,
            "variance": varianceData,
            "standard deviation": stdData,
            "max": maxData,
            "min": minData,
            "sum": sumData
        }
        return calculations


class UnitTests(unittest.TestCase):
    def test_calculate(self):
        # Test case 1: Input array [2,6,2,8,4,0,1,5,7]
        actual = MatrixStatisticalCalculation(np.array([2,6,2,8,4,0,1,5,7])).getStatisticalInfor()
        expected = {'mean': [[3.6666666666666665, 5.0, 3.0], [3.3333333333333335, 4.0, 4.333333333333333], 3.888888888888889], 'variance': [[9.555555555555557, 0.6666666666666666, 8.666666666666666], [3.555555555555556, 10.666666666666666, 6.222222222222221], 6.987654320987654], 'standard deviation': [[3.091206165165235, 0.816496580927726, 2.943920288775949], [1.8856180831641267, 3.265986323710904, 2.494438257849294], 2.6434171674156266], 'max': [[8, 6, 7], [6, 8, 7], 8], 'min': [[1, 4, 0], [2, 0, 1], 0], 'sum': [[11, 15, 9], [10, 12, 13], 35]}
        self.assertAlmostEqual(actual, expected, "Expected different output when calling 'getStatisticalInfo()' with '[2,6,2,8,4,0,1,5,7]'")

    def test_calculate2(self):
        # Test case 2: Input array [9,1,5,3,3,3,2,9,0]
        actual = MatrixStatisticalCalculation(np.array([9,1,5,3,3,3,2,9,0])).getStatisticalInfor()
        expected = {'mean': [[4.666666666666667, 4.333333333333333, 2.6666666666666665], [5.0, 3.0, 3.6666666666666665], 3.888888888888889], 'variance': [[9.555555555555555, 11.555555555555557, 4.222222222222222], [10.666666666666666, 0.0, 14.888888888888891], 9.209876543209875], 'standard deviation': [[3.0912061651652345, 3.39934634239519, 2.0548046676563256], [3.265986323710904, 0.0, 3.8586123009300755], 3.0347778408328137], 'max': [[9, 9, 5], [9, 3, 9], 9], 'min': [[2, 1, 0], [1, 3, 0], 0], 'sum': [[14, 13, 8], [15, 9, 11], 35]}
        self.assertAlmostEqual(actual, expected, "Expected different output when calling 'getStatisticalInfo()' with '[9,1,5,3,3,3,2,9,0]'")
    
    def test_calculate_with_few_digits(self):
        # Test case 3: Input array with fewer than 9 elements
        with self.assertRaisesRegex(ValueError, "Input Numpy Array must contain 9 numbers!!!"):
            MatrixStatisticalCalculation(np.array([2, 6, 2, 8, 4, 0, 1])).getStatisticalInfor()
        # self.assertRaisesRegex(ValueError, "Input Numpy Array must contain 9 numbers!!!", MatrixStatisticalCalculation(np.array([2,6,2,8,4,0,1])))
    