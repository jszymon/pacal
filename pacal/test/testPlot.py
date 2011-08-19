import unittest
from pylab import *
from scipy.integrate import quad, Inf
from numpy import *
from scipy.stats import *
from pacal import *
import time

import matplotlib.pyplot as plt

class TestPlot(unittest.TestCase):
    def setUp(self):
        print """====Test starting============================="""        
        self.N1 = NormalDistr(1,1)
        self.N2 = NormalDistr(2,1)
        self.SumN1N2 = self.N1 + self.N2
        self.Chi4 = ChiSquareDistr(3)
        self.U1 = UniformDistr(-4,-1)
        self. ts = time.time()
       
        
    def tearDown(self):
        te = time.time()
        print 'test done,   time=%7.5f s' % (te - self.ts)        
        
        
    def testPlotPdf(self):
        print """pdfs and histograms ..."""
        fig = plt.figure()
        self.Chi4.plot()
        self.SumN1N2.plot()
        self.U1.plot()
        self.Chi4.hist()
        self.SumN1N2.hist()
        self.U1.hist()
        self.assert_(True);
    def testHistdistr(self):
        print """histograms ..."""
        fig = plt.figure()
        self.Chi4.hist()
        self.SumN1N2.hist()
        self.U1.hist()
        self.assert_(True);
    
       
    def testDispBayesNetwork(self):
        print """3. display Bayes net of calculus"""
        C1 = ConstDistr(1)
        N1 = NormalDistr(0,1)
        N2 = NormalDistr(1,1)
        S1 = N1 + N2
        M1 = N1 / N1
        NegS1 = -S1
        self.assert_(True);

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPlot("testPlotPdf"))
    suite.addTest(TestPlot("testHistdistr"))
    suite.addTest(TestPlot("testDispBayesNetwork"))
    return suite;
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    plt.show();
