import unittest
from pylab import *
from scipy.integrate import quad, Inf
from numpy import *
from scipy.stats import *
from distr import *
from evaluator import *
from plotfun import *
import time

import matplotlib.pyplot as plt

class TestPlot(unittest.TestCase):
    def setUp(self):
        print """====Test starting============================="""        
        self.N1 = NormDistr(1,1)
        self.N2 = NormDistr(2,1)
        self.SumN1N2 = SumDistr(self.N1, self.N2)
        self.Chi4 = Chi2Distr(3)
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
        N1 = NormDistr(0,1)
        N2 = NormDistr(1,1)
        S1 = SumDistr(N1,N2)
        M1 = DivDistr(N1,N1)
        NegS1 = NegDistr(S1)    
        out= dispNet(MulDistr(NegS1,SumDistr(NegS1,NegDistr(M1))))
        print out
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
