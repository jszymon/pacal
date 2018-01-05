from __future__ import print_function

import unittest
from pylab import *
from scipy.integrate import quad, Inf
from numpy import *
from scipy.stats import *
from pacal import *
import time
import matplotlib.pyplot as plt
# sass  sa sa sas ala all jest to
class TestPlot(unittest.TestCase):
    def setUp(self):
        print("""====Test starting=============================""")
        self.N1 = NormalDistr(1, 1)
        self.N2 = NormalDistr(2, 1)
        self.SumN1N2 = self.N1 + self.N2
        self.Chi4 = ChiSquareDistr(3)
        self.U1 = UniformDistr(-4, -1)
        self. ts = time.time()


    def tearDown(self):
        te = time.time()
        print('test done,   time=%7.5f s' % (te - self.ts))


    def testPlotPdf(self):
        print("""pdfs and histograms ...""")
        fig = plt.figure()
        self.Chi4.plot()
        self.SumN1N2.plot()
        self.U1.plot()
        self.Chi4.hist()
        self.SumN1N2.hist()
        self.U1.hist()
        self.assertTrue(True);

    def testHistdistr(self):
        print("""histograms ...""")
        fig = plt.figure()
        self.Chi4.hist()
        self.SumN1N2.hist()
        self.U1.hist()
        self.assertTrue(True);



    def testBoxplot(self):
        pos = 1
        col = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'r', 'g', 'b', 'k', 'm', 'y', 'c']
        fig = plt.figure()
        for F in [NormalDistr(), UniformDistr(), CauchyDistr(), ChiSquareDistr(),
          ExponentialDistr(), BetaDistr(), ParetoDistr(), LevyDistr(), LaplaceDistr(),
          StudentTDistr(), SemicircleDistr(), FDistr(), DiscreteDistr()]:
          F.boxplot(pos, color=col[pos], useci=0.01, label=F.__class__.__name__)
          pos += 1
        legend()
        self.assertTrue(True);

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPlot("testBoxplot"))
    suite.addTest(TestPlot("testPlotPdf"))
    suite.addTest(TestPlot("testHistdistr"))

    return suite;
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    plt.show();
