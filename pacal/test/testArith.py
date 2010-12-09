from pylab import *;
from scipy.integrate import quad;
import basicstat as stat

from numpy import cos, arange, pi;
import unittest
from scipy.stats import *;
from pylab import *
from distr import * 

from plotfun import *
import time
import matplotlib.pyplot as plt

class TestArith(unittest.TestCase):
    def setUp(self):
        print """====Test starting============================="""        
        self. ts = time.time()
        
    def tearDown(self):
        te = time.time()
        print 'test done,   time=%7.5f s' % (te - self.ts)        
    
    def testCentralLimit(self):
        fig = plt.figure()
        n = 2
        mu = 2
        sigma = 1
        S = NormDistr(mu, sigma)
        for i in xrange(n-1):
            histdistr(S)
            S = InterpolatedDistr(SumDistr(S, NormDistr(mu,sigma)))                            
        plotdistr(S, -0, 30, 100)
        histdistr(S)
        print 'S=',dispNet(S)
        m, err = stat.mean(S)
        s, err = stat.std(S)
        #Y = InterpolatedDistr(S)
        print "error =", S.err, "interp. nodes used =", S.n_nodes, "#breakpoints =", len(S.breaks)        
        te = time.time()
        print te-self.ts
        self.assert_(abs(s-sqrt(n))<1e-15 and (te-self.ts)<1, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2})={3}, OR it should be faster time={4} s'.format(m,s,n,sqrt(n), (te - self.ts)))
        
    def testCentralLimitUniform(self):
        fig = plt.figure()
        n = 7
        a = 0
        b = 1
        sigma = 1
        S = UniformDistr(a, b)
        plotdistr(S, 0, 4, 1000)
        for i in xrange(n-1):
            histdistr(S)
            S = InterpolatedDistr(SumDistr(S, UniformDistr(a,b)))                            
            plotdistr(S, 0, 6, 1000)
        histdistr(S)
        print 'S=',dispNet(S)
        m, err = stat.mean(S)
        s, err = stat.std(S)
        #Y = InterpolatedDistr(S)
        print "error =", S.err, "interp. nodes used =", S.n_nodes, "#breakpoints =", len(S.breaks)        
        te = time.time()
        print te-self.ts
        print s - sqrt(n/12.0)
        #self.assert_(abs(s-sqrt(n))<1e-14 and (te-self.ts)<1, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2})={3}, OR it should be faster time={4} s'.format(m,s,n,sqrt(n), (te - self.ts)))
        self.assert_(abs(s-sqrt(n/12.0))<1e-14, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2}/12.0)={3}'.format(m,s,n,sqrt(n/12.0)))

    def testSumOfSquares(self):
        fig = plt.figure()
        n = 7
        mu=5
        sigma=1
        S=NormDistr(mu, sigma)
        for i in range(n-1):
            X = NormDistr(mu, sigma)
            S=SumDistr(S,X)
            histdistr(S)        
        print 'sum of {0} normal N({1}, {2}) variables:'.format(n, mu, sigma) 
        print dispNet(S)
        
    def testSumDependent(self):
        print """sum of two the same normal variables X+X=2X, 
        not two i.i.d."""
        fig = plt.figure()
        X = NormDistr(1,1)
        Y = SumDistr(X,X)
        correctAnswer =  NormDistr(2,2)
        plotdistr(Y)
        plotdistr(correctAnswer)
        histdistr(Y)
        m, err = stat.mean(Y)
        s, err = stat.std(Y)
        print dispNet(Y)
        print 'mean(X)={0}, std={1}, abs(std-sqrt(2))={2}'.format(m,s, abs(s-sqrt(2)))        
        self.assert_(abs(s-2)<1e-10,'suma zmiennych zaleznych jeszcze nie dziala, ale przynajmniej losowanie jest ok :)')

    def testCupOfTeaExample(self):
        """ The cup of tea example, take n times sip of tea 
            Y_0 = 100
            Y_1 = 100 - X_1, X_1 is drawn form [0, 100] 
            Y_2 = Y_1 - X_2, X_2 is drown form [0, Y_1] (Y_1.clone ?)     
            ...
            X_1, X_2, ... are dependent  
        """
        n = 2
        Y_0=ConstDistr(100)
        print dispNet(Y_0)
        self.assert_(1<0, 'not work yet')
    
def suite():
    suite = unittest.TestSuite()
    #suite.addTest(TestArith("testCentralLimit"))
    suite.addTest(TestArith("testCentralLimitUniform"))
    #suite.addTest(TestArith("testSumOfSquares"))
    #suite.addTest(TestArith("testSumDependent"))
    #suite.addTest(TestArith("testCupOfTeaExample"))
    return suite;

if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    show()
