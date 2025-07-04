from __future__ import print_function

from scipy.integrate import quad

import numpy as np
from numpy import cos, arange, pi
import unittest
from scipy.stats import *
from pacal import *

import time
import matplotlib.pyplot as plt

class TestArith(unittest.TestCase):
    def setUp(self):
        print("""====Test starting=============================""")
        self. ts = time.time()

    def tearDown(self):
        te = time.time()
        print('test done,   time=%7.5f s' % (te - self.ts))

    def testCentralLimit(self):
        #fig = plt.figure()
        n = 2
        mu = 2
        sigma = 1
        S = NormalDistr(mu, sigma)
        for i in range(n-1):
            #S.hist()
            S = S + NormalDistr(mu,sigma)
        #S.plot()
        #S.hist()
        m = S.mean()
        s = S.std()
        #print "error =", S.err, "interp. nodes used =", S.n_nodes, "#breakpoints =", len(S.breaks)
        te = time.time()
        assert np.abs(s-sqrt(n))<1e-15 and (te-self.ts)<10, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2})={3}, OR it should be faster time={4} s'.format(m,s,n,sqrt(n), (te - self.ts))

    def testCentralLimitUniform(self):
        #fig = plt.figure()
        n = 7
        a = 0
        b = 1
        sigma = 1
        S = UniformDistr(a, b)
        #S.plot()
        for i in range(n-1):
            #S.hist()
            S = S + UniformDistr(a,b)
            #S.plot()
        #S.hist()
        m = S.mean()
        s = S.std()

        #Y = InterpolatedDistr(S)
        #print("error =", S.err, "interp. nodes used =", S.n_nodes, "#breakpoints =", len(S.breaks))
        te = time.time()
        print(te-self.ts)
        print(s - sqrt(n/12.0))
        #self.assert_(abs(s-sqrt(n))<1e-14 and (te-self.ts)<1, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2})={3}, OR it should be faster time={4} s'.format(m,s,n,sqrt(n), (te - self.ts)))
        assert np.abs(s-sqrt(n/12.0))<1e-14, 'difference in comparison with theoretical std: mean(X)={0}, std={1}, sqrt({2}/12.0)={3}'.format(m,s,n,sqrt(n/12.0))

    def testSumOfSquares(self):
        #fig = plt.figure()
        n = 7
        mu=5
        sigma=1
        S = NormalDistr(mu, sigma)
        for i in range(n-1):
            X = NormalDistr(mu, sigma)
            S = S + X
            #S.hist()
        print('sum of {0} normal N({1}, {2}) variables:'.format(n, mu, sigma))

    # sum of two dependent variables would require different semantics
    #def testSumDependent(self):
    #    print("""sum of two the same normal variables X+X=2X,
    #    not two i.i.d.""")
    #    fig = plt.figure()
    #    X = NormalDistr(1,1)
    #    Y = X+X
    #    correctAnswer =  NormalDistr(2,2)
    #    Y.plot()
    #    correctAnswer.plot()
    #    Y.hist()
    #    m = Y.mean()
    #    s = Y.std()
    #    print('mean(X)={0}, std={1}, abs(std-sqrt(2))={2}'.format(m,s, abs(s-sqrt(2))))
    #    self.assertTrue(abs(s-2)<1e-10,'sum of dependent variables not working yet')

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
        self.assertTrue(0<1, 'not working yet - disable')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestArith("testCentralLimit"))
    suite.addTest(TestArith("testCentralLimitUniform"))
    suite.addTest(TestArith("testSumOfSquares"))
    #suite.addTest(TestArith("testSumDependent"))
    #suite.addTest(TestArith("testCupOfTeaExample"))
    return suite;

if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    plt.show()
