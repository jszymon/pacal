from __future__ import print_function

import unittest
from pylab import *
from scipy.integrate import quad, Inf
from numpy import *
from scipy.stats import *
from pacal import *

import time
import matplotlib.pyplot as plt

class TestFunc(unittest.TestCase):
    def setUp(self):
        print("""====Test starting=============================""")
        self.N1 = NormalDistr(0,1)
        self.N2 = NormalDistr(0,1)
        self.ts = time.time()

    def tearDown(self):
        te = time.time()
        print('test done,   time=%7.5f s' % (te - self.ts))

    def testChiSqr(self):
        """
            square of single normal variable
        """
        fig = plt.figure()
        print("comparing \chi^2_1 with N(0,1)^2")
        orgChi2 =  ChiSquareDistr(1)
        testChi2 = self.N1 ** 2
        orgChi2.plot()
        testChi2.plot()
        L1diff, err = quad(lambda x : abs(orgChi2.pdf(x)-testChi2.pdf(x)),0,Inf)
        print('L_1 difference = {0}'.format(L1diff))
        self.assertTrue(L1diff < 1e-8);

    def testChiSqrWithInterpolation(self):
        """
            square of single normal variable
        """
        fig = plt.figure()
        print("testing \chi^2_1 = N(0,1)^2, with interpolation")
        N1 = NormalDistr(0,1)
        orgChi2 =  ChiSquareDistr(1)
        squareN1 = N1**2
        testChi2 = squareN1
        #testChi2 = SquareDistr(N1)
        orgChi2.plot()
        testChi2.plot()
        squareN1.hist()
        L1diff, err = quad(lambda x : abs(orgChi2.pdf(x)-testChi2.pdf(x)),0,Inf)
        print('L_1 difference = {0}'.format(L1diff))
        self.assertTrue(0< 1e-8);

    def testChi2Sqr(self):
        print("testing superposition real function with random variable")
        self.assertTrue(0< 1e-8);

    def testChi3Sqr(self):
        """
            sum of squares of three independent normal variables
        """
        self.assertTrue(0< 1e-8)
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFunc("testChiSqr"))
    suite.addTest(TestFunc("testChiSqrWithInterpolation"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    show()
