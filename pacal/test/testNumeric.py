import random
import unittest
import time

from functools import partial

from numpy import sin, cos, pi, exp
from numpy import linspace
from numpy import finfo, double, unique, isnan, maximum, isfinite, diff

from pylab import figure, subplot, plot, legend
#from distr import *
from pacal.integration import *
from pacal.interpolation import ChebyshevInterpolator, ChebyshevInterpolator_PMInf
from pacal.interpolation import ChebyshevInterpolator_PInf, ChebyshevInterpolator_MInf

from pacal.segments import PiecewiseFunction
from pacal import *
from pacal.utils import chebt2, ichebt2, difffun


eps = finfo(double).eps


def cauchy(x, c=1.0):
    return c/(pi*(c**2+x*x))
def normpdf(x,mi=0,sigma=1):
    return 1.0/sqrt(2*pi)/sigma * exp(-(x - mi)**2/2/sigma**2)

def epsunique(tab, eps =1e-13):
    ub = unique(tab[isnan(tab)==False])
    return ub[~isfinite(ub) | hstack((True, (diff(ub)/maximum(1,abs(ub[1:])))>eps))]

class TestBasicstat(unittest.TestCase):

    def setUp(self):
        #print """====Test starting============================="""        
        self.tol = 1e-8
        self.errMsg = "tolerance ({0}) exceeded".format(self.tol)
        self.ts = time.time()
        
    def tearDown(self):
        te = time.time()
        #print 'mean(X)={0} (EX=df={2}), var={1} (VX=2*df={3})'.format(self.m, self.v, self.EX, self.VX)
        #print 'test done,   time=%7.5f s' % (te - self.ts)      
    
    def testNormal(self):
        #print """basic stats of normal distribution"""
        mu = -17.0
        sigma = 20.0
        self.X = NormalDistr(mu, sigma)
        self.m = self.X.mean()
        self.v = self.X.var()
        self.EX = mu
        self.VX = sigma**2
        self.assert_((abs(self.m-self.EX)+abs(self.v-self.VX)) < self.tol, self.errMsg);
        
    def testChi2(self):
        #print """basic stats of Chi2 distribution"""
        df = 50
        self.X = ChiSquareDistr(df)
        self.m = self.X.mean()
        self.v = self.X.var()
        self.EX = df
        self.VX = df*2
        self.assert_((abs(self.m-self.EX)+abs(self.v-self.VX)) < self.tol, self.errMsg);
        
    def testUniform(self):
        #print """basic stats of uniform distribution"""
        a = 0.0
        b = 6.0
        self.X = UniformDistr(a, b)
        self.m = self.X.mean()
        self.v = self.X.var()
        self.EX = (a+b)/2.0
        self.VX = (b-a)**2/12.0
        self.assert_((abs(self.m-self.EX) + abs(self.v-self.VX)) < self.tol, self.errMsg);

class TestIntegral(unittest.TestCase):
    def setUp(self):
        #print """====Test starting============================="""        
        self.tol = 8 * eps
        self.ts = time.time()
    def _assert_integ(self, err, I):
        self.assert_(err < self.tol, repr(I) + " +/- " + str(err))
    def tearDown(self):
        te = time.time()
        #print self.iClenshaw - self.exact, self.errClenshaw, self.tClenshaw - self.ts
        #print self.iQuad - self.exact, self.errQuad, self.tQuad - self.ts
        #print 'test done,   time=%7.5f s' % (te - self.ts)      
       
    def testClenshaw(self):
        #print """Test clenshaw integration"""
        exact = sin(2)
        I, errClenshaw = integrate_clenshaw(cos, 0, 2)
        self.tClenshaw = time.time()
        err = abs(I - exact)
        self._assert_integ(err, I)
    def testFejer2(self):
        #print """Test clenshaw integration"""
        exact = sin(2)
        I, errFejer2 = integrate_fejer2(cos, 0, 2)
        err = abs(I - exact)
        self._assert_integ(err, I)
    def testFejer2inv(self):
        # inverted interval
        exact = -sin(2)
        I, err = integrate_fejer2(cos, 2, 0)
        err = abs(I - exact)
        self._assert_integ(err, I)
    def testFejer2empty(self):
        # empty interval
        exact = 0
        I, errFejer2 = integrate_fejer2(cos, 1, 1)
        err = abs(I - exact)
        self._assert_integ(err, I)
    def testFejer2empty2(self):
        # empty interval, inexact bounds
        exact = 0
        I, errFejer2 = integrate_fejer2(cos, 0.1, 0.3/3)
        err = abs(I - exact)
        self._assert_integ(err, I)
    #def testCauchyPMInf(self):
    #    I, err2 = integrate_clenshaw_pminf(cauchy)
    #    err = abs(I - 1)
    #    #print err
    #    self.assert_(err < self.tol)
    def testCauchyPInf(self):
        I, err2 = integrate_fejer2_pinf(cauchy, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)
    def testCauchyMInf(self):
        I, err2 = integrate_fejer2_minf(cauchy, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)
    def testNormPMInf(self):
        I, err2 = integrate_clenshaw_pminf(normpdf)
        err = abs(I - 1)
        self._assert_integ(err, I)
    def testNormPInf(self):
        I, err2 = integrate_clenshaw_pinf(normpdf, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)
    def testNormMInf(self):
        I, err2 = integrate_clenshaw_minf(normpdf, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)
    def testNormPMInfFejer2(self):
        I, err2 = integrate_fejer2_pminf(normpdf)
        err = abs(I - 1)
        self._assert_integ(err, I)
    def testNormPInfFejer2(self):
        I, err2 = integrate_fejer2_pinf(normpdf, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)
    def testNormMInfFejer2(self):
        I, err2 = integrate_fejer2_minf(normpdf, 0)
        err = abs(I - 0.5)
        self._assert_integ(err, I)

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.tol = eps * 10
    def _str_error(self, err, interp):
        return 'error=' + str(err) + ", " + str(len(interp.Xs)) + " nodes"
    def get_max_diff(self, X, f1, f2):
        exact = f1(X)
        #Y = array([f2.interp_at(x) for x in X])
        Y = f2.interp_at(X)
        err = abs(exact - Y).max()
        return err
       
    def testBasicInterp(self):
        """Test basic Chebyshev interpolation on finite interval."""
        ci = ChebyshevInterpolator(cos, -2, 8)
        X = linspace(-2, 8, 1000)
        err = self.get_max_diff(X, cos, ci)
        self.assert_(err < self.tol, self._str_error(err, ci))
    def testPMInfInterp(self):
        def f(x):
            return exp(-x*x)
        ci = ChebyshevInterpolator_PMInf(f)
        X = exp(linspace(-10, 10, 1000))
        X = hstack([-X,X])
        err = self.get_max_diff(X, f, ci)
        self.assert_(err < self.tol, self._str_error(err, ci))
    def testCauchyPMInf(self):
        ci = ChebyshevInterpolator_PMInf(cauchy)
        X = exp(linspace(-10, 10, 1000))
        X = hstack([-X,X])
        err = self.get_max_diff(X, cauchy, ci)
        self.assert_(err < self.tol, self._str_error(err, ci))
    def testInterpPInf(self):
        def f(x):
            return exp(-x)
        ci = ChebyshevInterpolator_PInf(f, 2.0)
        X = ci.vt.L + exp(linspace(-10, 10, 1000))
        #print X
        err = self.get_max_diff(X, f, ci)
        self.assert_(err < self.tol, self._str_error(err, ci))
    def testInterpMInf(self):
        def f(x):
            return exp(x)
        ci = ChebyshevInterpolator_MInf(f, -2.0)
        X = ci.vt.U - exp(linspace(-10, 10, 1000))
        err = self.get_max_diff(X, f, ci)
        self.assert_(err < self.tol, self._str_error(err, ci))
    def testNormalPMInf(self):
        ci = ChebyshevInterpolator_PMInf(normpdf)
        X = exp(linspace(-10, 10, 1000))
        X = hstack([-X,X])
        err = self.get_max_diff(X, normpdf, ci)
        #print err, len(ci.Xs)
        self.assert_(err < self.tol, self._str_error(err, ci))
class TestVectorisation(unittest.TestCase):
    def setUp(self):
        #print """====Test starting============================="""        
        self.x = linspace(-2,2,10001)
        self.x_mat = (arange(25, dtype=float) / 25)#.reshape(5,5)
        self.xleft = linspace(-10,0,10000)
        self.xright = linspace(0,10,10000)
        self. ts = time.time()     
    def tearDown(self):
        te = time.time()
        #print 'test done,   time=%7.5f s' % (te - self.ts) 
    def testNormal(self):
        N1 = NormalDistr(0,1)
        y=N1.pdf(self.x)
        self.assert_(0 < 1)
    def testUniform(self):
        U1 = UniformDistr(0,1)
        y=U1.pdf(self.x)
        #print self.x,y
        #plot(self.x, y)
        self.assert_(0 < 1)    
    def testChi2(self):
        Ch = ChiSquareDistr(4)
        y=Ch.pdf(self.x)
        self.assert_(0 < 1)      
    def testUniformFor(self):
        U1 = UniformDistr(0,1)
        y1 = [U1.pdf(x) for x in self.x]
        #print self.x,y
        #plot(self.x, y)
        self.assert_(0 < 1)    
    def testChebFor(self):
        c = ChebyshevInterpolator(cauchy, -2, 2)
        y = [c.interp_at(x) for x in self.x]
        #print self.x,y
        #plot(self.x, y)
        self.assert_(0 < 1) 
    def testCheb(self):
        c = ChebyshevInterpolator(cauchy, -2, 2)
        y = c.interp_at(self.x)       
        #print self.x,y
        #plot(self.x, y)
        self.assert_(0 < 1) 
    def testChebMat(self):
        c = ChebyshevInterpolator(cauchy, -2, 2)
        y = c.interp_at(self.x_mat)       
        #print self.x_mat
        #print y
        self.assert_(0 < 1) 

class TestVarTransform(unittest.TestCase):
    def setUp(self):
        self.vt = VarTransformAlgebraic_PMInf()
    def testVarChangeWMask1(self):
        x, mask = self.vt.inv_var_change_with_mask(array([-1.0,0.0,1.0]))
        self.assert_(abs(x[1]) <= eps and all(mask == [False, True, False]))
    def testVarChangeWMask2(self):
        # test integer arrays
        x, mask = self.vt.inv_var_change_with_mask(array([-1,0,1]))
        self.assert_(abs(x[1]) <= eps and all(mask == [False, True, False]))
    def testVarChangeWMask3(self):
        # test scalars
        x, mask = self.vt.inv_var_change_with_mask(-1.0)
        self.assert_(mask == False)
    def testVarChangeWMask4(self):
        # test scalars
        x, mask = self.vt.inv_var_change_with_mask(1.0)
        self.assert_(mask == False)
    def testVarChangeWMask5(self):
        # test scalars
        x, mask = self.vt.inv_var_change_with_mask(0)
        self.assert_(abs(x) <= eps and mask == True)
    def testApplyWithTransform1(self):
        y = self.vt.apply_with_inv_transform(lambda x: x+1, array([-1,0,1]))
        self.assert_(all(abs(y - [0, 1, 0]) <= eps))
    def testApplyWithTransform2(self):
        y = self.vt.apply_with_inv_transform(lambda x: x+1, 0)
        self.assert_(abs(y - 1) <= eps)
    def testApplyWithTransform3(self):
        y = self.vt.apply_with_inv_transform(lambda x: x+1, 1, def_val = 3)
        self.assert_(abs(y - 3) <= eps)

class TestInterpolators(unittest.TestCase):
    def setUp(self):
        #print """====Test starting============================="""        
        self. ts = time.time()     
    def tearDown(self):
        te = time.time()
        print 'test done,   time=%7.5f s' % (te - self.ts) 
    def testChebcoef(self):
        S = PiecewiseFunction(fun=lambda x: sin(4*(x-0.5)), breakPoints=[-1, 1]).toInterpolated()
        seg = S.segments[0]
        Xs, Ys = seg.f.Xs, seg.f.Ys
        #print "Xs=", Xs
        #print "Ys=", Ys
        #print "Cs=", chebt2(Ys)
        #print "Ys=", ichebt2(chebt2(Ys))
        Cs = chebt2(Ys)
        YsStar= ichebt2(Cs)
        err = Ys- YsStar[-1::-1]
        print "Ys*-Ys", err
        figure()
        S.plot(color="r")
        D = S.diff()
        D.plot(color="k")
        #show()
        self.assert_(max(err) < 1e-14)
    def testTrim(self):
        S = PiecewiseFunction(fun=lambda x: sin(4*(x-0.5)), breakPoints=[-1, 1]).toInterpolated()
        I = S.trimInterpolators(abstol=1e-15)
        figure()
        subplot(211)
        S.plot(color="r")
        I.plot(color="k")
        r = S-I
        subplot(212)
        r.plot()
        #show()
        self.assert_(0 < 1)
    def testDiff(self):
        n=8
        S = UniformDistr(0,2)
        for i in range (n):
            S += UniformDistr(0,2)
        figure()
        S.plot(color='r')
        D = S.get_piecewise_pdf().diff()
        D.plot(color="k")
        for i in range (n):
            D = D.diff()
            D.plot(color="k")        
        #show()
    def testRoots(self):
        n=8
        S = UniformDistr(0,2)
        for i in range (n):
            S += UniformDistr(0,2)
        D = S.get_piecewise_pdf().diff().diff()
        
        r = D.roots()
        mi, xi = D.characteristicPoints()
        figure()
        D.plot()
        D.diff().plot()
        plot(xi,mi, 'bo')
        plot(r,D(r), 'ro')
        self.assert_(0 < 1)
    def testMinmax(self):
        n=8
        S = UniformDistr(0,2)
        for i in range (n):
            S += UniformDistr(0,2)
        D = S.get_piecewise_pdf()
        figure()
        for i in range (n):
            D=D.diff()
            maxi, xmaxi = D.max()
            mini, xmini = D.min()
            D.plot()
            plot(xmaxi,maxi,  'ko')
            plot(xmini,mini,  'ro')
        self.assert_(0 < 1)
    def testDiffInf(self):
        S = BetaDistr(3,4)* BetaDistr(3,4)
        S.summary()
        figure()
        S.plot()
        print S.get_piecewise_pdf()
        D = S.get_piecewise_pdf().diff()
        D.plot(color="k")
        show()
    def testDiffPdf(self):     
        #params.interpolation_asymp.debug_info = True   
        test = True
        for distr in [ChiSquareDistr(), NormalDistr(), UniformDistr(), CauchyDistr(), 
              ExponentialDistr(), BetaDistr(), ParetoDistr(), LevyDistr(), LaplaceDistr(),
              StudentTDistr(), SemicircleDistr(), FDistr(), BetaDistr(2.5,1.5), NormalDistr()]:
#        for distr in [NormalDistr()]:
            funcdf = distr.get_piecewise_cdf()
            
            funpdf = distr.get_piecewise_pdf()
            fun = funcdf.toInterpolated()
            ri  = distr.ci(1e-12)
            dw = fun.diff()
            dfun =  partial(difffun, fun, 1e-12)
            figure()
            x=linspace(ri[0], ri[1], 10000)
            y = dfun(x)
            err1 = funpdf(x)-y
            err2 = funpdf(x)-dw(x)
            plot(x,err1, color='r', label="complex dfiidiff")
            plot(x,err2, color='k', label="cheb. diff")
            #figure()
            #fun.plot(color='r', linewidth=6)
            #dw.plot(color='k', linewidth=2)
            print funpdf(x)
            print ":::::::::::", fun(array([-10, 10])+1e-8j)
            legend()
            max(abs(err1))<1e-8
            test = test and max(abs(err1))<1e-8
            test = test and max(abs(err2))<1e-8
            print fun
            print distr, max(abs(err1)), max(abs(err2))
            #assert(max(abs(err2))<1e-8)
            #assert(max(abs(err2))<1e-8)
        assert(test)
def suite():
    suite = unittest.TestSuite()
#    suite.addTest(TestInterpolators("testChebcoef"))
#    suite.addTest(TestInterpolators("testTrim"))
#    suite.addTest(TestInterpolators("testDiff"))
#    suite.addTest(TestInterpolators("testRoots"))
#    suite.addTest(TestInterpolators("testMinmax"))
#    suite.addTest(TestInterpolators("testDiffInf"))
    suite.addTest(TestInterpolators("testDiffPdf"))
#    suite.addTest(TestBasicstat("testNormal"))
#    suite.addTest(TestBasicstat("testChi2"))
#    suite.addTest(TestBasicstat("testUniform"))
#
#    suite.addTest(TestIntegral("testClenshaw"))
#    suite.addTest(TestIntegral("testFejer2"))
#    suite.addTest(TestIntegral("testFejer2inv"))
#    suite.addTest(TestIntegral("testFejer2empty"))
#    suite.addTest(TestIntegral("testFejer2empty2"))
#    ###suite.addTest(TestIntegral("testCauchyPMInf"))
#    suite.addTest(TestIntegral("testCauchyPInf"))
#    suite.addTest(TestIntegral("testCauchyMInf"))
#    #suite.addTest(TestIntegral("testNormPMInf"))
#    suite.addTest(TestIntegral("testNormPInf"))
#    suite.addTest(TestIntegral("testNormMInf"))
#    #suite.addTest(TestIntegral("testNormPMInfFejer2"))
#    suite.addTest(TestIntegral("testNormPInfFejer2"))
#    suite.addTest(TestIntegral("testNormMInfFejer2"))
#
#    suite.addTest(TestInterpolation("testBasicInterp"))
#    #suite.addTest(TestInterpolation("testPMInfInterp"))
#    #suite.addTest(TestInterpolation("testCauchyPMInf"))
#    suite.addTest(TestInterpolation("testInterpPInf"))
#    suite.addTest(TestInterpolation("testInterpMInf"))
#    #suite.addTest(TestInterpolation("testNormalPMInf"))
#    
#    suite.addTest(TestVectorisation("testNormal"))
#    suite.addTest(TestVectorisation("testUniform"))    
#    suite.addTest(TestVectorisation("testUniformFor"))
#    suite.addTest(TestVectorisation("testChi2"))
#    suite.addTest(TestVectorisation("testChebFor"))
#    suite.addTest(TestVectorisation("testCheb"))
#    suite.addTest(TestVectorisation("testChebMat"))
#    
#    suite.addTest(TestVarTransform("testVarChangeWMask1"))
#    suite.addTest(TestVarTransform("testVarChangeWMask2"))
#    suite.addTest(TestVarTransform("testVarChangeWMask3"))
#    suite.addTest(TestVarTransform("testVarChangeWMask4"))
#    suite.addTest(TestVarTransform("testVarChangeWMask5"))
#    suite.addTest(TestVarTransform("testApplyWithTransform1"))
#    suite.addTest(TestVarTransform("testApplyWithTransform2"))
#    suite.addTest(TestVarTransform("testApplyWithTransform3"))

    return suite;
   
if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    show()

    #unittest.main()
