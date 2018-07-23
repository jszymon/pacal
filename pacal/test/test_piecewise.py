from __future__ import print_function

import unittest

import operator

from pylab import show, subplot, pi
from numpy import linspace, multiply, add, divide
from numpy import unique, isnan, isscalar, diff
from numpy import Inf, sign, isinf, isfinite, exp, log, cos, sin
from numpy import logspace, sqrt, log10

from pylab import plot, semilogx, semilogy, xlabel, ylabel, axis, legend, hist

from pacal.utils import epsunique, estimateDegreeOfPole
from pacal.utils import findinv

#from testNumeric import cauchy, normpdf
import time
import matplotlib.pyplot as plt
import matplotlib.text as text
plt.rcParams.update({'figure.max_open_warning': 0})

from pacal.segments import *
from pacal.indeparith import *
del testPole
from pacal.integration import integrate_fejer2
from math import factorial


from scipy.optimize import fmin


def cauchy(x, c=1.0, ex=2.0):
    return c/(pi*(c**2+abs(x)**ex))
def prodcauchy(x, c=1.0, ex=2.0):
    return c**2/(pi**2 * (x - c**2) * (x + c**2)) * 2 * log ((abs(x)/c**2))
def prodNuniform01(x, n=1):
    if n==1:
        return 1.0+zeros(size(x))#*(0.0<=x)*(x<=1.0)
    else:
        return log(1/x) ** (n-1)/factorial(n-1)
def normpdf(x,mu=0,sigma=1):
    return 1.0/sqrt(2*pi)/sigma * exp(-(x - mu)**2/2.0/sigma**2)
def stable05pdf(x, mu =0.0, c=1.0):
    if isscalar(x):
        x=array([x])
    y = zeros_like(x)
    ind = x>0
    if any(x):
        y[ind] = sqrt(c/(2*pi))/(x[ind]-mu)**1.5 * exp(-0.5 * c / (x[ind]-mu))
    return y
def chisqr(x, k=1):
    assert 0<k<100
    coeffs = [0,2.506628274631001,2,2.506628274631,4,7.519884823893001,16,37.59942411946501,96,263.1959688362551,768,2368.763719526296,7680,26056.40091478925,92160,338733.2118922602,1290240,5080998.178383904,20643840,86376969.03252636,371589120,1641162411.618001,7431782400,34464410643.97802,163499212800,792681444811.4906,3923981107200,19817036120287.35,102023508787200,535059975247760.3,2856658246041600,1.551673928218494e+016,8.5699747381248e+016,4.810189177477336e+017,2.742391916199936e+018,1.587362428567526e+019,9.324132515079782e+019,5.55576849998637e+020,3.356687705428722e+021,2.055634344994941e+022,1.275541328062914e+023,8.016973945480292e+023,5.102165312251657e+024,3.28695931764694e+025,2.142909431145696e+026,1.413392506588176e+027,9.428801497041062e+027,6.360266279646801e+028,4.337248688638937e+029,2.989325151434023e+030,2.081879370546656e+031,1.464769324202674e+032,1.040939685273327e+033,7.470323553433536e+033,5.412886363421323e+034,3.959271483319787e+035,2.922958636247489e+036,2.17759931582587e+037,1.636856836298603e+038,1.24123161002075e+039,9.49376965053202e+039,7.32326649912245e+040,5.696261790319202e+041,4.467192564464701e+042,3.531682309997874e+043,2.814331315612744e+044,2.260276678398662e+045,1.829315355148299e+046,1.491782607743107e+047,1.225641287949337e+048,1.014412173265315e+049,8.456924886850515e+049,7.100885212857138e+050,6.004416669663804e+051,5.112637353257176e+052,4.38322416885467e+053,3.783351641410374e+054,3.287418126640914e+055,2.875347247471884e+056,2.531311957513567e+057,2.242770853028042e+058,1.99973644643572e+059,1.794216682422407e+060,1.619786521612925e+061,1.47125767958639e+062,1.344422812938723e+063,1.235856450852555e+064,1.142759390997911e+065,1.062836547733212e+066,9.942006701681694e+066,9.352961620052071e+067,8.848385964496985e+068,8.417665458046964e+069,8.052031227692019e+070,7.744252221403107e+071,7.488389041753833e+072,7.279597088118913e+073,7.113969589665893e+074,6.988413204594182e+075,6.900550501975964e+076,6.848644940502514e+077]
    return 1.0/coeffs[k] * x**(k/2.0-1.0) * exp(-x/2.0)
def betapdf(x, alpha=2, beta=2):
    return x**(alpha-1) * (1-x)**(beta-1)

# helper functions to replace lambdas
class theoretic_hlp1:
    def __init__(self, theoretic):
        self.theoretic = theoretic
    def __call__(self, x):
        return self.theoretic(x, 1)
def inv_x(x): return 1.0 / x
def inv_x_sq(x): return 1.0 / (x**2)
def f_zero(x): return x*0 + 0.0
def f_one(x): return x*0 + 1.0
def f_half(x): return x*0 + 0.5
def f_2_3(x): return x*0 + 2.0/3.0
def cauchy_10_2(x): return cauchy(x, 10.0, 2.0)
def normpdf_1(x): return normpdf(x, 1.0)
def minvlog(x): return -1.0/log(x)
def f_sq(x): return x*x
def f_pole_helper(x): return 1.0/sqrt(2.0)/1.772453850905516/sqrt(x)*exp(-x/2.0)
def f_helper1(x): return 1.0/4.5*(3-x)
def f_helper2(x): return 2772.0*betapdf(x-1,alpha = 6, beta=6)
def f_helper3(x,k): return normpdf(x,0,sqrt(k))
def f_helper4(x,k): return stable05pdf(x,0,k**2)
def neg_chisq(x, n): return chisqr(-x, n)
def f_4mx(x): return 4.0-x
def m_half_log_abs(x): return -0.5*log(abs(x))

def do_testWithTheoretical(n = 1, op = conv, f = None, theoretic = None, comp_w_theor = True, a = 0, b=Inf, isPoleAtZero = True, splitPoints=[],
                        plot_tails = False, asympf = None):
    """Universal comarision with theoretical distribution
        a,b only -inf -1,0,1,inf allowed and a*b>=0"""
    Lexp = -5
    Uexp = 2
    Npts = 10000

    if f is None:
        X = array([])
        f=PiecewiseFunction([])
        if a ==- Inf and -1 <= b:
            f.addSegment(MInfSegment(-1.0, theoretic_hlp1(theoretic)))
            X = -logspace(Uexp, 0, Npts);
        if a<=-1 and 0<=b:
            X = concatenate([X, -logspace(0,Lexp, Npts)])
            if isPoleAtZero:
                f.addSegment(SegmentWithPole(-1.0, 0.0, theoretic_hlp1(theoretic), left_pole = False))
            else:
                f.addSegment(Segment(-1.0, 0.0,  theoretic_hlp1(theoretic)))
        if a<=0 and 1<=b:
            X = concatenate([X, logspace(Lexp,0, Npts)])
            if isPoleAtZero:
                f.addSegment(SegmentWithPole(0.0, 1.0, theoretic_hlp1(theoretic)))
            else:
                f.addSegment(Segment(0.0, 1.0,  theoretic_hlp1(theoretic)))
        if a<=1 and b==Inf:
            X = concatenate([X, logspace(0, Uexp, Npts)])
            f.addSegment(PInfSegment(1.0, theoretic_hlp1(theoretic)))
        assert(len(f.segments)>0)
        if len(splitPoints)>0:
            f = f.splitByPoints(splitPoints)
    else:
        X = linspace(a, b, Npts)

    g=f;
    g=f.toInterpolated()
    h=g;
    ints = zeros(n)
    LInferr = zeros(n)
    LInferr2 = zeros(n)

    for i in range(n):
        if comp_w_theor:
            subplot(n,2,2*i+1)
            h.plot(color = 'g', linewidth = 3)
            Y = theoretic(X,i+1)
            plot(X, h(X), color='r',linewidth=3)
            plot(X, Y, color='k')
            subplot(n,2,2*i+2)
            plot(X, abs(h(X) - Y)/Y)
            print("iii====", i, h)
            #for seg in h.segments:
            #    figure()
            #    loglog(seg.f.getNodes()[0], seg.f.getNodes()[1], "o")
            #    figure()
            ind = where(Y!=0)
            LInferr[i] = max(abs(h(X) - Y))
            LInferr2[i] = max(abs(h(X[ind]) - Y[ind])/Y[ind])
        else:
            subplot(n,1,i+1)
            h.plot(color = 'g', linewidth = 3)
            plot(X, h(X), color='r',linewidth=2)
            print("iii====", i, h)
            LInferr[i] = 0
            LInferr2[i] = 0


        ints[i] = h.integrate()
        print("int error=", 1.0-ints[i])
        if i==n-1:
            break
        print(h,g)
        h=op(h, g)
    if plot_tails:
        plt.figure()
        if asympf is not None:
            h.plot_tails(asympf = asympf)
        else:
            h.plot_tails()
    print(ints)
    print("interr = ", 1-ints)
    print("LInfAbs = ", LInferr)
    print("LInfRel = ", LInferr2)
    return ints

def ratioTester(f, Lexp = -10, Uexp=1):
    """Universal comarision with theoretical distribution
        a,b only -inf -1,0,1,inf allowed and a*b>=0"""
    Lexp = -5
    Uexp = 2
    Npts = 10000

    subplot(3,1,1)
    f.plot(color='k')
    print("f=", f, f(array([1.0])))
    r=convdiv(f,f)
    print("r=", r)

    r.plot(color='k', linewidth=2)
    subplot(3,1,2)
    p = convprod(r,r)
    print("p=", p)
    d = convdiv(r,r)
    print("d=", d)
    p.plot(color='k', linewidth=1, linestyle='-')
    d.plot(color='r', linewidth=1, linestyle='-')
    subplot(3,1,3)
    err = d - p
    err.plot()

    intf = f.integrate()
    intr = r.integrate()
    intp = p.integrate()
    intd = d.integrate()
    interr = err.integrate()

    print("intf = ", intf, 1-intf)
    print("intr = ", intr, 1-intr)
    print("intp = ", intp, 1-intp)
    print("intd = ", intd, 1-intd)
    print("interr = ", interr)
    return max(1-intp, 1-intd)

def inversionTester(f, Lexp = -10, Uexp=1):
    """Universal comarision with theoretical distribution
        a,b only -inf -1,0,1,inf allowed and a*b>=0"""
    Lexp = -5
    Uexp = 2
    Npts = 10000

    subplot(3,1,1)
    f.plot(color='k')
    r= f.copyComposition(inv_x, inv_x, inv_x_sq)
    p=convprod(f,r);
    d=convdiv(f,f);

    print("f=", f)
    print("r=", r)
    r.plot(color='k', linewidth=2)

    subplot(3,1,2)
    print("p=", p)
    print("d=", d)
    p.plot(color='k', linewidth=1, linestyle='-')
    d.plot(color='r', linewidth=1, linestyle='-')
    subplot(3,1,3)
    err = d - p
    err.plot(color='r')

    intf = f.integrate()
    intr = r.integrate()
    intp = p.integrate()
    intd = d.integrate()
    interr = err.integrate()

    print("intf = ", intf, 1-intf)
    print("intr = ", intr, 1-intr)
    print("intp = ", intp, 1-intp)
    print("intd = ", intd, 1-intd)
    print("interr = ", interr)
    return max(1-intp, 1-intd)

def integrationTester(fun):
    ifun1 = fun.cumint()
    ifun = ifun1.toInterpolated()
    print(fun)
    print(ifun1)
    print(ifun)
    subplot(2,1,1)
    fun.plot()
    subplot(2,1,2)
    ifun.plot()
    subplot(2,1,1)
    #ifun.hist()
    levels =  array([0.05, 0.1, 0.5, 0.9, 0.95])
    for level in levels:
        print("level={0} cv={1}".format(level, ifun.inverse(level)))
    print("levels=",levels, "cv=", ifun.inverse(levels))

def f_compl(x): return 1.0 - (x-1.0)
def f_sin(x): return sin(pi/2.0*x)
def f_sqm2(x): return (x-2)**2
def f_exp_sqm2(x): return exp(-(x-2)**2)
def f_m2x(x): return -2*x
def f_2x(x): return 2*x
def f_cubrt(x): return x**(1.0/3)
def f_cub(x): return x*x*x
def f_cub_der(x): return 3*x*x
def sqrrrootdistr(x, _n = 1):
    if isscalar(x):
        if x <= 0 or x > 1:
            y = 0
        else:
            y = 1.0 / sqrt(x)
    else:
        mask = (x > 0) & (x <= 1)
        y = zeros_like(asfarray(x))
        y[mask] = 1.0 / sqrt(x[mask])
    return y / 2
def sqrrrootdistr2(x, _n = 1):
    if isscalar(x):
        if x < -1 or x > 1:
            y = 0
        else:
            y = 1.0 / sqrt(abs(x))
    else:
        mask = (x >= -1) & (x <= 1)
        y = zeros_like(asfarray(x))
        y[mask] = 1.0 / sqrt(abs(x[mask]))
    return y / 4
def logdistr(x, _n = 1):
    if isscalar(x):
        if x < -1 or x > 1:
            y = 0
        else:
            y = abs(log(abs(x)))
    else:
        x = asfarray(x)
        mask = (x >= -1) & (x <= 1)
        y = zeros_like(x)
        y[mask] = abs(log(abs(x[mask])))
    return y
def logdistr2(x, n = 1):
    return logdistr(x, n) / 2

class TestPicewiseConvs(unittest.TestCase):
    def setUp(self):
        #print """====Test starting============================="""
        self.n = 3
        self.nseg = 5
        self.tol = 1e-13
        self.ts = time.time()
    def tearDown(self):
        te = time.time()
        print('test done,   time=%7.5f s' % (te - self.ts))
    def testPiecewiseFunction(self):
        """Fragment of Cauchy distribution as piecewise function"""
        fig = plt.figure()
        f=PiecewiseFunction([])
        seg1 = Segment(0.0, 1.0, f_sin)
        seg2 = Segment(1.0, 2.0, f_compl)
        seg3 = DiracSegment(2.0, 0.25)
        #seg4 = Segment(2.0, 3.0, lambda x: exp(-(x-2)**2))
        seg4 = Segment(2.0, 3.0, f_sqm2)
        seg5 = PInfSegment(3.0, f_exp_sqm2)
        f.addSegment(seg1)
        f.addSegment(seg2)
        f.addSegment(seg3)
        f.addSegment(seg4)
        f.addSegment(seg5)
        subplot(3,1,1)
        f.plot(linewidth=1, color = 'k', linestyle='-')
        f=f.toInterpolated()
        subplot(3,1,2)
        f.plot(linewidth=1, color = 'k', linestyle='-')
        #x = linspace(-self.nseg/2.0-2, self.nseg/2.0+2,1001)
        #fun = lambda x : cauchy(x)
        #f=PiecewiseFunction([])
        #for i in range(self.nseg) :
        #    seg = Segment(i - self.nseg/2.0, i + 1 - self.nseg/2.0, fun)
        #    f.addSegment(seg)
        #print f
        #f.plot(linewidth=3, color = 'b', linestyle='-')
        subplot(3,1,3)
        g=conv(f,f)
        g.addSegment(DiracSegment(4.0, 0.0625))
        g.plot(linewidth=1, color = 'k', linestyle='-')
        #finterp = f.toInterpolated()
        #finterp.plot(linewidth=1, color = 'k', linestyle='-')
        gint = g.integrate()
        fint = f.integrate()
        print("f=",f)
        print("g=",g)
        print(fint + fint, gint, fint * fint - gint)
        self.assertTrue(0 < 1)

    def testConvUniformMix(self):
        """Sum of N mixtures of uniform random variables"""
        fig = plt.figure()
        segf1 = ConstSegment(0.0,1.0, 0.5)
        segf2 = ConstSegment(1.0,2.0, 0.0)
        segf3 = ConstSegment(2.0, 3.0, 0.5)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        f.addSegment(segf3)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(self.n):
            h = conv(h,f)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            print(i, h)
        self.f = h
        int = h.integrate()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvUniform(self):
        """Sum of N U[0,1] random variables"""
        plt.figure()
        segf1 = ConstSegment(0.0,1.0, 1)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        h=f
        h.plot()
        for i in range(self.n) :
            h = conv(h,f)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            print(i, h)
        int = h.integrate()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvCauchy(self):
        """Mean of the N Cauchy random variables, on figure difference between original single cauchy and mean of N ..."""
        segf1 = Segment(-1.0, 0.0, cauchy)
        segf2 = Segment(0.0, 1.0, cauchy)
        segf3 = MInfSegment(-1.0, cauchy)
        segf4 = PInfSegment(1, cauchy)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        f.addSegment(segf3)
        f.addSegment(segf4)
        h=f
        #fig = plt.figure()
        #h.plot(linewidth=3, color = 'b', linestyle='-')
        int = h.integrate()
        print(int)
        fig = plt.figure()
        print(h)
        print(h.integrate())
        for i in range(self.n):
            h = conv(h,f)
            #h.plot(color = 'r', linewidth=i+1)
            #k = h.copyShiftedAndScaled(0.0,1.0*(i+2))
            k = h
            subplot(self.n,1,i+1)
            #k.plot(linewidth=1, color = 'k', linestyle='-')
            X = linspace(-10000,10000,10000)
            E = k(X) - cauchy(X,c=i+2)#f(X)
            plot(X, E, linewidth=0.5, color = 'k', linestyle='-')
            figure()
            X = linspace(-10,10,10000)
            E = k(X) - cauchy(X,c=i+2)#f(X)
            plot(X, E, linewidth=0.5, color = 'k', linestyle='-')
            print(i, h)
        I = h.integrate()
        figure()
        h.plot_tails(asympf = f_m2x)
        self.assertTrue(abs(I-1)<self.tol, 'integral = {0}'.format(abs(I-1)))

    def testConvXalpha(self):
        """Mean of the N Cauchy random variables, on figure difference between original single cauchy and mean of N ..."""
        fig = plt.figure()
        segf11 = Segment(-1.0, 0.0, cauchy_10_2)
        segf12 = Segment(0.0, 1.0,  cauchy_10_2)
        segf2 = MInfSegment(-1.0,   cauchy_10_2)
        segf3 = PInfSegment(1,      cauchy_10_2)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf11)
        f.addSegment(segf12)
        f.addSegment(segf3)
        f.plot(linewidth=1, color = 'r', linestyle='-')

        #h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(1):
            h = conv(f,f)
            k = h.copyShiftedAndScaled(0.0,1.0*(0+2))
            #subplot(self.n,1,i+1)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            f.plot(linewidth=1, color = 'r', linestyle='-')
            k.plot(color = 'b')
            fig = plt.figure()
            estimateDegreeOfPole(f, Inf)
            estimateDegreeOfPole(h, Inf)
        print(f)
        print(h)
        it = h.integrate()
        intf = f.integrate()
        print("intf=", intf * intf, "int=", it, ", tol=", intf*intf-it)
        self.assertTrue(abs(it-1)<self.tol, 'integral = {0}'.format(abs(it)))

    def testConvNormalScaled(self):
        """Mean of the N normal random variables, on figure difference between original single cauchy and mean of N ..."""
        fig = plt.figure()
        segf1 = MInfSegment(0.0,  normpdf_1)
        segf2 = PInfSegment(2.0,  normpdf_1)
        segf3 = Segment(0.0, 2.0, normpdf_1)
        #segf4 = Segment(-1.0,  0.0, lambda x:normpdf(x, mu=1))
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        #f.addSegment(segf4)
        f.addSegment(segf3)
        h=f
        ts = time.time()
        for i in range(self.n):
            h= conv(h,f)
            #h= convmean(h,f, p=0.5, q=0.5)
            subplot(3,1,1)
            h.plot(linewidth=1, linestyle='-')
            print(i, h)
            #int = h.integrate()
            #print 'initegral=', int
        #subplot(3,1,2)
        #k = h.copyShiftedAndScaled(float(self.n+1),sqrt(self.n+1))
        #k.plot(linewidth=1, linestyle='-')
        #subplot(3,1,3)
        ###print mean(h)
        ###print std(h)
        k = h
        x=linspace(-100,100,10000)
        y=abs(k(x)-normpdf(x, i+2, sqrt(i+2)))
        y = k(x)
        figure()
        semilogy(x,y,linewidth=0.5, linestyle='-')
        x=linspace(0,4,10000)
        y=abs(k(x)-normpdf(x, i+2, sqrt(i+2)))
        figure()
        semilogy(x,y,linewidth=0.5, linestyle='-')
        fig = figure()
        Xs, Ys = k.segments[-1].f.getNodes()
        Xs = Xs[1:-1]
        Ys = Ys[1:-1]
        Ys = log(Ys)
        Ys -= log(normpdf(Xs, i+2, sqrt(i+2)))
        plot(Xs,Ys,"bo")
        Xs, Ys = k.segments[0].f.getNodes()
        Xs = Xs[1:-1]
        Ys = Ys[1:-1]
        Ys = log(Ys)
        Ys -= log(normpdf(Xs, i+2, sqrt(i+2)))
        plot(Xs,Ys,"bo")
        fig.gca().set_xlim(-100, 100)

        #figure()
        #h.plot_tails(asympf = lambda x: -0.25*exp(x)*exp(x) + 0.5*exp(x))
        int = h.integrate()
        #figure()
        #h.plot_tails()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvStable0_5(self):
        """Sum of stable distributions with alpha=0.5."""
        segf1 = Segment(0.0, 2.0, stable05pdf)
        segf2 = PInfSegment(2.0, stable05pdf)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        h=f

        h.plot(linewidth=1, linestyle='-')
        figure()
        n =2

        me = zeros(n+1)
        va = zeros(n+1)

        #me[0] = f.mean()
        #va[0] = f.var()
        ints = zeros(n);
        for i in range(n):
            h= conv(h,f)
            #h= convmean(h,f, p=0.5, q=0.5)
            subplot(n,1,1+i)
            me[i+1] = h.mean()
            va[i+1] = h.var()
            h.plot(linewidth=1, linestyle='-')
            print(i, h)
            int = h.integrate()
            print('initegral=', int, repr(int), abs(int-1))
        #print("mean=", me)
        #print("var=", va)
        figure()
        h.plot_tails()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))


    def testConvprodUniform(self):
        """Product of the N U[0.4, 1.4] random variables"""
        fig = plt.figure()
        segf1 = ConstSegment(0.4,1.4, 1)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(self.n) :
            h = convprod(h,f)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            print(i, h)
            int = h.integrate()
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvprodUniform01(self):
        """Product of the N U[0.0, 1.0] random variables"""
        fig = plt.figure()
        segf1 = ConstSegment(0.0,0.5, 1.0)
        segf2 = ConstSegment(0.5,1.0, 1.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        n = 5
        int = zeros(n)
        relerr = zeros(n)
        for i in range(n) :
            h = convprod(h,f)
            plt.subplot(2,1,1)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            X  = logspace(-10,0,1000)
            Y = prodNuniform01(X,i+2)
            plot(X,Y,linewidth=1, color = 'r', linestyle='--')
            plt.subplot(2,1,2)
            plot(X, abs(Y-h(X))/Y, linewidth=1, color = 'b', linestyle='-')
            print(i, h)
            int[i] = h.integrate()
            relerr[i] = max(abs(Y-h(X))/Y)
            print('initegral=', int)
            print('initerr=', relerr)
        self.assertTrue(abs(max(int)-1)<self.tol, 'integrals = {0}'.format(abs(int-1)))

    def testConvprodUniform2(self):
        """Product of U[1, 2] and U[-1,1]"""
        fig = plt.figure()
        segf1 = ConstSegment(1.0,2.0,1.0)
        segg1 = ConstSegment(-1,1,0.5)
        segg2 = ConstSegment(-1,0,0.5)
        segg3 = ConstSegment(0.00,1.00,1.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        #g.addSegment(segg1) # TODO: this should work
        #g.addSegment(segg2)
        g.addSegment(segg3)

        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        g.plot(linewidth=3, color = 'm', linestyle='-')
        for i in range(1) :
            h = convprod(g,h)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            print(i, h)
            int = h.integrate()
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvprodMix(self):
        """Product of N mixtures of uniform random variables"""
        plt.figure()
        segf1 = ConstSegment(0.1, 1.1, 1.0/5.0)
        segf3 = ConstSegment(1.1, 2.1, 4.0/5.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        #f.addSegment(segf2)
        f.addSegment(segf3)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(self.n) :
            h = convprod(h,f)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvprodCauchy(self):
        """Product of two Cauchy random variables"""
        fig = plt.figure()
        #segf1 = MInfSegment(-10.0, lambda x:cauchy(x, 1))
        #segf2 = PInfSegment(10.0, lambda x:cauchy(x))
        #segf3 = Segment(-0.1, 0, lambda x:cauchy(x))
        #segf4 = Segment(0, 0.1, lambda x:cauchy(x))
        #segf5 = Segment(-10.0, -0.1, lambda x:cauchy(x))
        #segf6 = Segment(0.1, 10.0, lambda x:cauchy(x))
        #f = PiecewiseFunction([])
        #f.addSegment(segf2)
        #f.addSegment(segf1)
        #f.addSegment(segf3)
        #f.addSegment(segf4)
        #f.addSegment(segf5)
        #f.addSegment(segf6)
        #h=f
        segf1 = MInfSegment(-1.0,  cauchy)
        segf2 = PInfSegment(1.0,   cauchy)
        segf3 = Segment(-1.0, 0.0, cauchy)
        segf4 = Segment(0.0, 1.0,  cauchy)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf3)
        f.addSegment(segf4)
        h=f
        subplot(3,1,1)
        h.plot(linewidth=1, color = 'b', linestyle='-')
        for i in range(1) :
            h = convprod(h,f)
            k = convdiv(h,f)
            subplot(3,1,2)
            h.plot(linewidth=1, color = 'r', linestyle='-')
            k.plot(linewidth=1, color = 'k', linestyle='-')
            inth = h.integrate()
            intk = h.integrate()
            print(inth, h)
            print(intk, k)
            X=linspace(-100,100, 100000 )
            Y=h(X)-k(X)
            plot(X,Y,'k')

            subplot(3,1,3)
        self.assertTrue(abs(max(inth,intk)-1)<self.tol, 'integral = {0}'.format(max(inth,intk)-1))
    def testConvprodCauchyUni(self):
        """Product of Cauchy and uniform variables"""
        fig = plt.figure()
        segf1 = MInfSegment(-1.0, cauchy)
        segf2 = PInfSegment(1.0, cauchy)
        segf3 = Segment(-1.0, 0.0, cauchy)
        segf4 = Segment(0.0, 1.0, cauchy)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf3)
        f.addSegment(segf4)
        segg1 = ConstSegment(-1.0, 0.0, 0.5)
        segg2 = ConstSegment(0.0, 1.0, 0.5)
        g = PiecewiseFunction([])
        g.addSegment(segg1)
        g.addSegment(segg2)
        h=f
        subplot(3,1,1)
        h.plot(linewidth=1, color = 'b', linestyle='-')
        for i in range(1) :
            h = convprod(h,g)
            #g2= g.copyComposition(lambda x: 1.0/x, lambda x: 1.0/x, lambda x: 1.0/x**2 )
            #h = convdiv(h,g2)
            k = convdiv(h,g)
            subplot(3,1,2)
            h.plot(linewidth=1, color = 'r', linestyle='-')
            subplot(3,1,3)
            k.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvprodMix2(self):
        """Another product of N mixtures of uniform random variables"""
        plt.figure()
        segf1 = ConstSegment( -1.5, -0.5, 1)
        segf2 = ConstSegment( -1.0, -0.0, 0.3)
        segf3 = ConstSegment( 0.0, 1.0, 0.7)
        #segf2 = MInfSegment(  -0.0, lambda x : 1.0* exp(x))
        #segf2 = PInfSegment(  0.0, lambda x : 0.7* exp(-x))
        f = PiecewiseFunction([])
        g = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf3)
        g.addSegment(segf1)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        g.plot(linewidth=3, color = 'g', linestyle='-')
        #print "left_degree=", estimateDegreeOfPole(h,-Inf)
        #print "left_degree=", estimateDegreeOfPole(h,Inf)
        for i in range(1):
            print(i)
            h = convprod(g,h)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
            #fig = plt.figure()
            #print "left_degree=", estimateDegreeOfPole(h,-Inf)
            #print "left_degree=", estimateDegreeOfPole(h,Inf)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvProdNormal(self):
        """Product two normal random variables"""
        ffig = plt.figure()
        segf1 = MInfSegment(-1.0, normpdf)
        segf2 = PInfSegment(1.0, normpdf)
        segf3 = Segment(-1.0, -0.0, normpdf)
        segf4 = Segment(0.0,  1, normpdf)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf4)
        f.addSegment(segf3)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(1) :
            h = convprod(h,f)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    @unittest.skip("loss of accuracy")
    def testConvProdSqrtUni(self):
        """Product two normal random variables"""
        ffig = plt.figure()
        #segf1 = Segment(0.0, 1.0, lambda x:3.0/1.0*x**1.0 * (1-x))
        n=2.0
        #segf1 = Segment(0.0, 1.0, lambda x:(n+1)/(n) * x ** (1/n))
        #segf1 = Segment(0.0, 2.0, lambda x:pi/2 * sqrt(1 - (x-1) ** 2))
        #segf1 = Segment(0.0, 1.0, lambda x: exp(-1/x))
        segf1 = Segment(0.0, 0.5,  minvlog)
        segf3 = Segment(0.0, 1.0,  f_one)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        g.addSegment(segf3)

        h=convprod(g,f);
        f.plot(linewidth=1, color = 'r', linestyle='-')
        g.plot(linewidth=1, color = 'b', linestyle='-')
        h.plot(linewidth=1, color = 'k', linestyle='-')
        int = h.integrate();
        fint = f.integrate()
        gint = g.integrate()
        axis((0,2,0,5))
        print(">>>>>>>>>>>>>>",fint, int, gint*fint-int)
        print(h)
        self.assertTrue(abs(fint-int)<self.tol, 'integral = {0}'.format(abs(gint*fint-int)))

    @unittest.skip("needs review")
    def testConvDivInf1(self):
        """Product two normal random variables"""
        #segf1 = Segment(0.0, 1.0, lambda x:3.0/1.0*x**1.0 * (1-x))
        n=4
        f = PiecewiseFunction([])
        for i in range(n):
            if i>=0:
                segf0 = Segment(2**i - 1.0/2**(i+2) , 2**i + 1.0/2**(i+2) , f_one)
                f.addSegment(segf0)
            if i<n:
                segfi = Segment(2**i + 1.0/2**(i+2) , 2**(i+1) - 1.0/2**(i+3) , f_zero)
                f.addSegment(segfi)
        segf0 = Segment(2**(-1), 1 - 1.0/2**(2) , f_zero)
        f.addSegment(segf0)

        #f.plot();
        h=convdiv(f,f);

        g= f.copyComposition(inv_x, inv_x, inv_x_sq)
        h2=convprod(f,g);

        #h=convprod(f,f); # no poles in this case

        fig = plt.figure()
        subplot(2,1,1)
        f.semilogx(linewidth=2, color = 'k', linestyle='-')
        axis((0.1, 10, 0, 1.1))
        plt.title('$f_4$')
        subplot(2,1,2)

        f.semilogx(linewidth=2, color = 'k', linestyle='-')
        axis((0.1, 10, 0, 1.1))
        plt.title('$f_4$')
        #h.semilogx(linewidth=1, color = 'k', linestyle='-')03,
        #X=logspace(-2,3,10000)
        #Y=h(X);
        #fig = plt.figure()
        #semilogx(X,Y)
        int = h.integrate()
        fint = f.integrate()
        print(">>>>>>>>>>>>>>",fint, int, fint*fint-int)
        print(i)
        print(h)
        fig = plt.figure()
        h.semilogx(linewidth=1, color = 'b', linestyle='-')
        h2.semilogx(linewidth=1, color = 'k', linestyle='-')
        axis((0.08, 12, 0, 2.1))
        plt.title('$h_4 = f_4 \oslash g_4$, ${h_4}^\prime = f_4 \odot g_4$')

        fig = plt.figure()
        h.semilogx(linewidth=1, color = 'b', linestyle='-')
        for o in fig.findobj(text.Text):
            o.set_fontsize(14)
        axis((0.08, 12, 0, 2.1))
        plt.title('$h_4 = f_4 \oslash g_4$')

        fig = plt.figure()
        subplot(2,1,1)
        f.semilogx(linewidth=2, color = 'k', linestyle='-')
        axis((0.1, 10, 0, 1.1))
        subplot(2,1,2)
        g.semilogx(linewidth=2, color = 'k', linestyle='-')
        for o in fig.findobj(text.Text):
            o.set_fontsize(14)
        axis((0.1, 10, 0, 70.0))
        plt.ylabel('$g_4 = 1\oslash f_4$', fontsize=20)

        subplot(2,1,1)
        plt.ylabel('$f_4$', fontsize=20)

        fig = plt.figure()
        h2.semilogx(linewidth=1, color = 'k', linestyle='-')
        axis((0.08, 12, 0, 2.1))
        plt.title('$f_4 \odot g_4$')

        fig = plt.figure()
        subplot(2,1,1)
        h.semilogx(linewidth=1, color = 'b', linestyle='-')
        h2.semilogx(linewidth=1, color = 'k', linestyle='-')
        axis((0.08, 12, 0, 2.1))
        subplot(2,1,2)
        xi = linspace(0.08,12,10000)
        yi = abs(h(xi) - h2(xi))
        loglog(xi,yi)
        for o in fig.findobj(text.Text):
            o.set_fontsize(14)
        plt.ylabel('$\log|h_4-h_4^\prime|$', fontsize=20)
        axis((0.08, 12, 1e-19, 1.5e-14))

        subplot(2,1,1)
        plt.ylabel('$h_4 = f_4 \oslash g_4$, ${h_4}^\prime = f_4 \odot g_4$', fontsize=20)


        #h.plot(linewidth=1, color = 'k', linestyle='-')
        #fig = plt.figure()
        #f.semilogx(linewidth=2, color = 'k', linestyle='-')
        self.assertTrue(abs(fint-int)<self.tol, 'integral = {0} != {1} (absdiff={2})'.format(fint, int, abs(fint-int)))

    def testConvProdInf1(self):
        """Product two normal random variables"""
        fig = plt.figure()
        #segf1 = Segment(0.0, 1.0, lambda x:3.0/1.0*x**1.0 * (1-x))
        n=4
        f = PiecewiseFunction([])
        for i in range(n):
            if i>=0:
                segf0 = Segment(2**i - 1.0/2**(i+2) , 2**i + 1.0/2**(i+2) , f_one)
                f.addSegment(segf0)
            if i<n:
                segfi = Segment(2**i + 1.0/2**(i+2) , 2**(i+1) - 1.0/2**(i+3) , f_zero)
                f.addSegment(segfi)
        #segf0 = Segment(2**n - 1.0/2**(n+2) , 2**(n) + 1.0/2**(n+2) , lambda x: 0.0*x + 1)
        #f.addSegment(segf0)
        segf0 = Segment(2**(-1), 1 - 1.0/2**(2) , f_zero)
        f.addSegment(segf0)


        #f.plot();
        g= f.copyComposition(inv_x, inv_x, inv_x_sq)
        h=convprod(f,g);
        f.semilogx(linewidth=2, color = 'k', linestyle='-')
        g.semilogx(linewidth=2, color = 'b', linestyle='-')
        #h.plot(linewidth=1, color = 'k', linestyle='-')
        #X=logspace(-2,3,10000)
        #Y=h(X);
        #fig = plt.figure()
        #semilogx(X,Y)
        fig = plt.figure()
        h.semilogx(linewidth=1, color = 'k', linestyle='-')
        #h.semilogx()
        int = h.integrate()
        fint = f.integrate()
        print(">>>>>>>>>>>>>>",fint, int, fint*fint-int)
        print(h)
        self.assertTrue(abs(fint*fint-int)<self.tol, 'integral = {0}'.format(abs(fint*fint-int)))

    def testConvDivUni2(self):
        """Product two normal random variables"""
        fig = plt.figure()
        seg1 =  Segment( 0.0, 1.0,  f_half)
        seg2 =  Segment(-1.0, 0.0,  f_half)
        f = PiecewiseFunction([])
        f.addSegment(seg2)
        f.addSegment(seg1)

        g=f
        g.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(2) :
            g = convdiv(g,f)
            g.plot(linewidth=1, color = 'k', linestyle='-')
            plt.figure()
            plt.title('tails i={0}'.format(i))
            g.plot_tails()
            int = g.integrate()
            print(i, g)
            print('initegral=', int)
        axis((1e-10,10,1e-10,0.51))
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvDivNormal(self):
        """Quotient two normal random variables"""
        fig = plt.figure()
        segf1 = MInfSegment(-1.0, normpdf)
        segf2 = PInfSegment(1.0, normpdf)
        segf3 = Segment(-1.0, 0, normpdf)
        segf4 = Segment(0, 1.0, normpdf)
        #segf5 = Segment(-3.0, -1.0, lambda x:normpdf(x))
        #segf6 = Segment(1.0, 3.0, lambda x:normpdf(x))
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf3)
        f.addSegment(segf4)

        segg1 = MInfSegment(-1.0, cauchy)
        segg2 = PInfSegment(1.0, cauchy)
        segg3 = Segment(-1.0, 1.0, cauchy)
        g = PiecewiseFunction([])
        g.addSegment(segg1)
        g.addSegment(segg2)
        g.addSegment(segg3)
        #f.addSegment(segf5)
        #f.addSegment(segf6)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        h = convdiv(h,f)
        figure()
        h.plot_tails(asympf = f_m2x)
        h.plot(linewidth=1, color = 'k', linestyle='-')
        k = h - g
        int = h.integrate()
        print(h)
        print('initegral=', int)
        fig = plt.figure()
        g.plot(color = 'b')
        h.plot(color = 'k')
        fig = plt.figure()
        k.plot(color = 'r')
        plt.title("Quotient two normal random variables - difference between theoretical Cauchy distribution")
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvDivCauchy(self):
        """Quotient of two cauchy random variables"""
        fig = plt.figure()
        segf1 = MInfSegment(-10.0,   cauchy)
        segf2 = PInfSegment(10.0,    cauchy)
        segf3 = Segment(-1.0, 0,     cauchy)
        segf4 = Segment(0, 1.0,      cauchy)
        segf5 = Segment(-10.0, -1.0, cauchy)
        segf6 = Segment(1.0, 10.0,   cauchy)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf3)
        f.addSegment(segf4)
        f.addSegment(segf5)
        f.addSegment(segf6)
        h=f
        #h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(1) :
            h = convdiv(h,f)

            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvDivUni(self):
        """Quotient two normal random variables"""
        ffig = plt.figure()
        segf1 = ConstSegment(1.0, 3.0, 1.0/2.0)
        segg1 = ConstSegment(-2.0, 0.0, 1.0/3.0)
        segg2 = ConstSegment(0.0, 1.0, 1.0/3.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        g.addSegment(segg1)
        g.addSegment(segg2)
        h=f
        h.plot(linewidth=3, color = 'b', linestyle='-')
        g.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(3) :
            h = convdiv(f,g)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            #k = convdiv(f,h)
            #k.plot(linewidth=1, color = 'b', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        axis((-10,10,0,0.51))
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    @unittest.skip("loss of accuracy")
    def testConvDivUniWide(self):
        """Quotient two random variables.

        Segments are close to zero causing a very wide but finite result."""
        e = 1e-3
        segf1 = ConstSegment(1.0, 3.0, 1.0/2.0)
        segg1 = ConstSegment(-2.0, -e, 0.5 / (2.0-e))
        segg2 = ConstSegment(0.0, e, 0.5/e)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        g.addSegment(segg1)
        g.addSegment(segg2)
        h=f
        #ffig = plt.figure()
        #h.plot(linewidth=3, color = 'b', linestyle='-')
        #g.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(1) :
            h = convdiv(h,g)
            figure()
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testCauchy(self):
        """Quotient of two cauchy random variables"""

        fig = plt.figure()
        segf1 = MInfSegment(-2.0, cauchy)
        segf2 = PInfSegment(2.0, cauchy)
        segf3 = Segment(-0.5, 0, cauchy)
        segf4 = Segment(0, 0.5, cauchy)
        segf5 = Segment(-2.0, -0.5, cauchy)
        segf6 = Segment(0.5, 2.0, cauchy)
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        f.addSegment(segf3)
        f.addSegment(segf4)
        f.addSegment(segf5)
        f.addSegment(segf6)
        h=f
        #h.plot(linewidth=3, color = 'b', linestyle='-')
        h = convdiv(f,f)
        k = convprod(f,f)
        X=logspace(-10,10,10000)
        X = concatenate([-logspace(10,-10,10000), X])
        yc = prodcauchy(X)
        Yh = abs((h(X) - yc)/yc)
        Yk = abs((k(X) - yc)/yc)
        plot(X, Yh, linewidth=1, color = 'k', linestyle='-')
        plot(X, Yk, linewidth=1, color = 'b', linestyle='-')
        plt.figure()
        fig = plt.figure()
        h.plot(linewidth=1, color = 'k', linestyle='-')
        plot(X, prodcauchy(X), color = 'g', linestyle='-', linewidth=2)
        fig = plt.figure()
        k.plot(linewidth=1, color = 'k', linestyle='-')
        plot(X, prodcauchy(X), color = 'g', linestyle='-', linewidth=2)
        print("int(h)=", 1.0-h.integrate())
        print("int(k)=", 1.0-k.integrate())
        print("h=", h)
        print("k=", k)
        print('difference=', max(Yh), max(Yk))
        self.assertTrue(abs(max(Yh))<1e-9, 'diff = {0}'.format(abs(max(max(Yh), max(Yh)))))

    def testConvDivMix(self):
        """Quotient of N mixtures of uniform random variables"""
        segf1 = ConstSegment( 0.0, 1.0, 1.0/5.0)
        segf2 = ConstSegment( -2.0, 0.0, 2.0/5.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        h=f
        #h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(self.n):
            h = convdiv(h,f)
            fig = plt.figure()
            h.plot(linewidth=2, color = 'k', linestyle='-')
            ## fig = plt.figure()
            ## h.plot_tails()
            ## # plot vartransformed right tail
            ## fig = plt.figure()
            ## I = h.segments[-1].f
            ## T = linspace(I.vt.var_min, I.vt.var_max, 1000)
            ## print I.vt.var_min, I.vt.var_max
            ## #print I.Xs
            ## #print I.Ys
            ## print "ws=", I.weights
            ## T = logspace(-30, 0, 1000)
            ## Y = I.transformed_interp_at(T)
            ## Y2 = I.f(T)
            ## print "WWW", I.transformed_interp_at(1e-50), I.f(1e-50)
            ## print I.f, h.segments[0].f.f
            ## I3 = ChebyshevInterpolator(I.f, I.vt.var_min, I.vt.var_max)
            ## Y3 = I3(T)
            ## #for i, x in enumerate(I.Xs):
            ## #    print i, x, I.Ys[i], I.f(x)
            ## #plot(T, Y)
            ## loglog(T, Y)
            ## loglog(T, Y2)
            ## loglog(T, Y3, "r")
            figure()
            h.plot_tails()
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        figure()
        h.plot_tails()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvMin(self):
        """Max of N of uniform random variables"""
        fig = plt.figure()
        segf1 = ConstSegment(0.0, 1.0, 0.5)
        segf2 = ConstSegment(1.0, 2.0, 0.5)
        segf3 = ConstSegment(1.0, 2.0, 2.5/5.0)
        segf4 = ConstSegment(2.0, 3.0, 2.5/5.0)
        f = PiecewiseFunction([])
        h = PiecewiseFunction([])

        f.addSegment(segf1)
        f.addSegment(segf2)
        h.addSegment(segf3)
        h.addSegment(segf4)
        h=f;
        f.plot(linewidth=3, color = 'r', linestyle='-')
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(2) :
            h = convmin(f,h)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral - 1 =', int-1)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))
    def testConvmaxNormal(self):
        """Mean of the N normal random variables, on figure difference between original single cauchy and mean of N ..."""
        fig = plt.figure()
        segf1 = MInfSegment(-2.0, normpdf)
        segf2 = PInfSegment(2.0, normpdf)
        segf3 = Segment(-2.0, 2.0, normpdf)
        #segf4 = Segment(-1.0,  0.0, lambda x:normpdf(x, mu=1))
        f = PiecewiseFunction([])
        f.addSegment(segf2)
        f.addSegment(segf1)
        #f.addSegment(segf4)
        f.addSegment(segf3)
        h=f
        ts = time.time()
        subplot(2,1,1)
        f.plot()
        for i in  range(3):
            h.plot(linewidth=1, linestyle='-')
            h= convmin(h,f)
            #h= convmean(h,f, p=0.5, q=0.5)
            subplot(2,1,2)
            int = h.integrate()
            print(i, h)
            print('initegral=', int)
        subplot(2,1,2)
        h.plot(linewidth=1, linestyle='-')
        #subplot(3,1,3)
        #print mean(h)
        #print std(h)
        #x=linspace(-100,100,10000)
        #y=abs(k(x)-normpdf(x))
        #plot(x,y,linewidth=0.5, linestyle='-')
        int = h.integrate()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvMax(self):
        """Min of N of uniform random variables"""
        fig = plt.figure()
        segf1 = ConstSegment(0.0, 1.0, 0.4)
        segf2 = ConstSegment(1.0, 2.0, 0.6)
        segf3 = ConstSegment(0.0, 1.0, 1.0/3.0)
        segf4 = ConstSegment(1.0, 2.0, 2.0/3.0)
        f = PiecewiseFunction([])
        h = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        #h.addSegment(segf3)
        #h.addSegment(segf4)
        h=f;
        f.plot(linewidth=3, color = 'r', linestyle='-')
        h.plot(linewidth=3, color = 'b', linestyle='-')
        for i in range(2) :
            h = convmax(f,h)
            h.plot(linewidth=1, color = 'k', linestyle='-')
            int = h.integrate()
            print(i, h)
            print('initegral - 1 =', int-1)
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvDiscrete(self):
        """Sum of N of binomial random variables"""

        segf1 = DiracSegment(0.0, 0.3)
        segf2 = DiracSegment(1.0, 0.7)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        print(f.getDiracs())
        for i in range(3):
            print(i)
            f=convdiracs(f, f, fun = operator.add)
            print("cumsum=", f.integrate()-1, f)
        plt.figure()
        f.plot();
        for i in range(3):
            f=convdiracs(f, f, fun = operator.sub)
            print("cumsum=", f.integrate()-1, f)
        plt.figure()
        f.plot();
    def testConvDiscreteMix1(self):
        """Sum of N of binomial random variables"""
        print("===============")
        segf1 = DiracSegment(0.0, 0.3)
        segf2 = DiracSegment(0.5, 0.5)
        segf3 = ConstSegment(0.0, 0.5, 0.4)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        f.addSegment(segf3)
        g = PiecewiseFunction([])
        #seg1 = Segment(-1.0, 0.0, lambda x: 1.0*x+1.0)
        #seg2 = Segment(0.0, 1.0, lambda x: 1.0 - 1.0*x)
        seg3 = ConstSegment(0.0, 1.0, 0.5)
        seg2 = DiracSegment(1.0, 0.5)
        #seg4 = ConstSegment(2.0, 3.0, 1.0/2.0)
        g.addSegment(seg3)
        g.addSegment(seg2)
        n=3
        subplot(n+1,1,1)
        f.plot(linewidth=2, color = 'b');
        g.plot(linewidth=2, color = 'k');

        axis((-1,8,0,1.1))
        print("===", f, f.integrate())
        print("===", g, g.integrate())

        for i in range(n):
            print(i)
            g=conv(f, g)
            subplot(n+1,1,i+2)
            g.plot(linewidth=2, color = 'k')
            print("cumsum=", g.integrate()-1, g)
            int = g.integrate()
            print(i, g)
            print('initegral - 1 =', int-1)
        plt.figure()
        c =g.cumint()
        c.plot()
        plt.figure()
        g.plot()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    def testConvDiscreteMix2(self):
        """Sum of N of binomial random variables"""
        print("===============")
        segf1 = DiracSegment(1.0, 0.3)
        segf2 = DiracSegment(0.0, 0.2)
        segf3 = ConstSegment(1.0, 2.0, 0.5)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        f.addSegment(segf3)
        g = PiecewiseFunction([])
        #seg1 = Segment(-1.0, 0.0, lambda x: 1.0*x+1.0)
        seg2 = Segment(1.0, 2.0, f_compl)
        seg3 = ConstSegment(0.0, 1.0, 0.2)
        seg4 = DiracSegment(0.0, 0.1)
        seg5 = DiracSegment(1.0, 0.2)
        #seg4 = ConstSegment(2.0, 3.0, 1.0/2.0)
        g.addSegment(seg2)
        g.addSegment(seg3)
        g.addSegment(seg4)
        g.addSegment(seg5)
        n=1
        subplot(n+3,2,1)
        f.plot(linewidth=2, color = 'b');
        #g.plot(linewidth=2, color = 'k');
        subplot(n+3,2,2)
        #f.plot(linewidth=2, color = 'b');
        g.plot(linewidth=2, color = 'k');

        axis((-1,8,0,1.1))
        print("===", f, f.integrate())
        print("===", g, g.integrate())
        h=g
        p=g
        for i in range(n):
            print(i)
            subplot(n+3,2,2*i+3)
            h=convmin(h, f)
            h.plot(linewidth=2, color = 'k')
            subplot(n+3,2,2*i+4)
            p=convmax(p, f)
            p.plot(linewidth=2, color = 'k')

            inth = h.integrate()
            intp = p.integrate()
            print(i, " =========================")
            print("cumsum=", inth-1, inth, h)
            print("cumsum=", intp-1, intp, p)
        subplot(n+3,2,2*n+3)
        ch =h.cumint().toInterpolated()
        ch.plot()

        subplot(n+3,2,2*n+4)
        cp =p.cumint().toInterpolated()
        cp.plot()

        subplot(n+3,2,2*n+5)
        h.plot()

        subplot(n+3,2,2*n+6)
        p.plot()

        err = max(abs(inth-1),abs(inth-1))
        self.assertTrue( err < self.tol, 'integral = {0}'.format(err))

    def testRatio(self):
        """Product and quotient of ratios of identical distributions
        should be the same."""
        segf1 = ConstSegment(0.5, 1.0, 2.0/3.0)
        segf2 = ConstSegment(1.0, 2.0, 2.0/3.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        f.addSegment(segf2)
        #f = conv(f, f)
        plt.figure()
        subplot(3,1,1)
        r1 = convdiv(f, f)
        r1.semilogx(color = 'k',)
        q = convdiv(r1, r1)
        p = convprod(r1, r1)
        #plt.figure()
        subplot(3,1,2)
        #q.semilogx(color = 'k',)
        p.semilogx(color = 'r',)
        q.semilogx(color = 'k',)
        X=logspace(log10(1.0/16.0),log10(16.0), 100000 )
        Y=p(X)-q(X)
        subplot(3,1,3)
        plot(X,Y,'k')
        pint = p.integrate()
        qint = q.integrate()
        print(pint, qint, pint-qint)

    def testSegmentsWithPoles(self):
        """Poles..."""
        #fun1 =  lambda x: x**2.5+1.0/abs(x) ** 0.5
        #fun1 =  lambda x: 1.0/sqrt(2.0)/1.772453850905516/sqrt(x)*exp(-x/2.0)
        #fun2 =  lambda x: 1.0/sqrt(2.0)/1.772453850905516/sqrt(x)*exp(-x/2.0)
        fun1 = f_pole_helper
        fun2 = f_pole_helper
        #fun2 =  lambda x: exp(-x)/sqrt(abs(x))
        segf1 = InterpolatedSegmentWithPole( 0.0, 1.0, fun1, residue = 0.0, exponent=1.5)
        segf2 = InterpolatedSegmentWithPole( 0.0, 1.0, fun2, residue = 1.0/sqrt(2.0)/1.772453850905516, exponent=0.5)
        f = PiecewiseFunction([])
        g = PiecewiseFunction([])
        f.addSegment(segf1)
        g.addSegment(segf2)
        #f = conv(f, f)
        plt.figure()
        print("est=", estimateDegreeOfPole(fun1, 0.0))

        plt.figure()
        f.plot(linewidth=2)
        X=logspace(-3,-0.0001,100000)
        Y = fun1(X)
        plot(X,Y)

        plt.figure()
        X=logspace(-10,-0.0001,10000)
        Y = fun1(X)
        Yf = f(X)
        semilogx(X, (Yf -Y), 'b')
        semilogx(X, (Yf -Y)/Y, 'r')

        plt.figure()
        g.plot(linewidth=2)

        plt.figure()
        #g.plot()
        X=logspace(-10,-0.00001,10000)
        Y = fun2(X)
        Yf = g(X)

        semilogx(X, (Yf - Y), 'b')
        semilogx(X, (Yf - Y)/Y, 'r')
        print(f.integrate());
        print(g.integrate());
        print(integrate_fejer2(fun1, 0.0, 1.0));
        print(integrate_fejer2(fun2, 0.0, 1.0));
        #X=-1 + logspace(0,-3,100)
        #Y = fun2(X)
        #plot(X, Y, 'r')
    def testSquareNormal(self):
        """Quotient two normal random variables"""
        fun = f_pole_helper
        segf1 = MInfSegment(-1.0, normpdf)
        segf2 = PInfSegment(1.0,  normpdf)
        segf3 = Segment(-1.0, 0,  normpdf)
        segf4 = Segment(0, 1.0,   normpdf)
        #segf5 = Segment(-3.0, -1.0, lambda x:normpdf(x))
        #segf6 = Segment(1.0, 3.0, lambda x:normpdf(x))
        f1 = PiecewiseFunction([])
        f2 = PiecewiseFunction([])
        f1.addSegment(segf2)
        f1.addSegment(segf1)
        f1.addSegment(segf3)
        f1.addSegment(segf4)
        print(f1)
        g1 = f1.copySquareComposition()
        #g2 = f2.copySquareComposition()
        plt.figure(1)
        f1.plot();
        #f2.plot();
        g1.plot()
        #g2.plot()
        x1=logspace(-3,1,10000)
        y1 = fun(x1)
        plot(x1,y1,'g',linewidth = 5.0)
        g1.plot()
        plt.figure(2)
        estimateDegreeOfPole(g1,0)
        print("g1=", g1, g1(2))
        plt.figure(1)
        h = g1.toInterpolated()
        #print h
        #h.plot()
        #h = k, # h=g1 wted kwadat jest liczony bez interpolacji
        #h=g1
        n=2;
        for i in range(n):
            plt.figure(3)
            h.plot(color = 'g', linewidth = 4)
            xx = logspace(-10,2,10000)
            yy = chisqr(xx,i+1)
            plot(xx,yy, color='k')
            plt.figure(4)
            subplot(n,1,i+1)
            semilogx(xx, (abs(h(xx) - yy)))
            ylabel('abs. error')
            print("iii====", i, h)
            int = h.integrate()
            print("int error=", 1.0-int)
            if i==n-1:
                break
            h=conv(h,g1)

        #print "tu problem dla Xs=1 jednen z koncow dotyka bieguna:\n", seg.f.getNodes()

    def testLogNotOneOverX(self):
        "Distribution which is O(1/x) whose log is not O(1/x)."
        f1 = PiecewiseFunction([])
        for i in range(1,148):
            seg = ConstSegment((i+1.0) - 1.0/(i), i+1.0, 1.0/(i+1))
            f1.addSegment(seg)
        fig = plt.figure()
        g = f1.copyComposition(log, exp, exp )
        f2 = PiecewiseFunction([])
        for i in range(1,25):
            seg = ConstSegment((i+1.0) - 1.0/(i), i+1.0, 1.0/(i+1))
            f2.addSegment(seg)

        g2 = f2.copyComposition(sqrt, f_sq, f_2x)  # this is O(1/x)
        f3 = PiecewiseFunction([])
        for i in range(1,125):
            seg = ConstSegment((i+1.0) - 1.0/(i), i+1.0, 1.0/(i+1))
            f3.addSegment(seg)
        g3 = f3.copyComposition(f_cubrt, f_cub, f_cub_der )  # this is O(1/x)
        #g = f.copyComposition(lambda x: 1.0/x, lambda x: 1.0/x, lambda x: -1.0/x/x )  # this is O(1/x)
        #g.plot()
        xi=linspace(1,5,1000)
        yi=1.0/(xi)
        subplot(4,1,1)
        f2.semilogx(linewidth=1, color = 'k')

        subplot(4,1,2)
        g.semilogx(color = 'k')
       # plot(xi,0*xi+2,linestyle = '--',linewidth = 2.0)

        subplot(4,1,3)
        g2.semilogx(color = 'k', linestyle='-',linewidth = 1.0)
        plot(xi,2*yi, color = 'c',linestyle = '--')

        subplot(4,1,4)
        g3.semilogx(color = 'k', linestyle='-',linewidth = 1.0)
        plot(xi,3*yi, color = 'c',linestyle = '--')
        for o in fig.findobj(text.Text):
            o.set_fontsize(14)
        subplot(4,1,1)
        plt.ylabel('$X$', fontsize=20)

        subplot(4,1,1)
        plt.ylabel('$\log X$', fontsize=20)
        subplot(4,1,1)
        plt.ylabel('$\sqrt{X}$', fontsize=20)
        subplot(4,1,1)
        plt.ylabel('$\sqrt[3]{X}$', fontsize=20)


        #g.semilogx(color = 'k', linestyle='-')
        #g2.semilogx(color = 'b', linestyle='--')
        #g3.semilogx(color = 'r', linestyle='-.')

        fig = plt.figure()
        subplot(4,1,1)
        xi=linspace(1,25,1000)
        yi=1.0/(xi)

        plot(xi,yi, color = 'b',linestyle = '--', label = '1/x')
        legend()
        f2.plot(linewidth=2, color = 'k')

        xi=linspace(1,5,1000)
        yi=1.0/(xi)
        axis((0, 25, 0, 1.0))
        subplot(4,1,4)
        g.plot(color = 'k', linewidth = 2.0)
        #plot(xi,0*xi+2,linestyle = '-',linewidth = 1.0)

        subplot(4,1,2)
        plot(xi,2*yi, color = 'b',linestyle = '--', label = '2/x')
        legend()

        #plt.legend('envelope K/x')
        g2.plot(color = 'k', linestyle='-',linewidth = 2.0)
        subplot(4,1,3)
        plot(xi,3*yi, color = 'b',linestyle = '--', label = '3/x')
        legend()
        #plt.legend('envelope K/x'))
        g3.plot(color = 'k', linestyle='-',linewidth = 2.0)
        for o in fig.findobj(text.Text):
            o.set_fontsize(12)
        subplot(4,1,1)
        plt.ylabel('$X$', fontsize=16)
        subplot(4,1,4)
        plt.ylabel('$\log(X)$', fontsize=16)
        subplot(4,1,2)
        plt.ylabel('$\sqrt{X}$', fontsize=16)
        subplot(4,1,3)
        plt.ylabel('$\sqrt[3]{X}$', fontsize=16)
        print(g2)
        print(f1.integrate())
        print(g.integrate())
        print(g2.integrate())
        print(g3.integrate())
    @unittest.expectedFailure
    def testConvmeanUniform(self):
        """Weighted mean of two uniform random variables
         with different variance, optimal solution"""
        plt.figure()
        segf1 = ConstSegment(0.0, 1.0, 1.0/1.0)
        segf2 = Segment(0.0, 3.0, f_helper1)#lambda x: 1.0/4.5*(3-x))
        #segf2 = Segment(0.0, 6.0, lambda x: 1.0/6.0 +0.0*x)
        segf3 = ConstSegment(-1.0, 2.0, 1.0/3.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        g.addSegment(segf3)
        #h.plot()
        #mf = f.mean()
        #vf = f.var()
        #mg = g.mean()
        #vg = g.var()
        f.plot(linewidth=1, color = 'r', linestyle='-')
        g.plot(linewidth=1, color = 'r', linestyle='-')
        #int = h.integrate()
        #mh = h.mean()
        #vh = h.var()
        #print(mf, vf, mg, vg)

        w0=[0.5, 0.5]
        fun1 = lambda w : convmean(f,g, p=w[0],q=w[1]).var();
        fun2 = lambda w : convmean(f,g, p=w[0],q=w[1]).iqrange(0.49);
        fun3 = lambda w : convmean(f,g, p=w[0],q=w[1]).medianad();
        print(fun1(w0), fun2(w0), fun3(w0))
        w = fmin(fun1, w0)
        w2 = fmin(fun2, w0)
        w3 = fmin(fun3, w0)
        w=abs(w)/(abs(w[1])+abs(w[0]))
        w2=abs(w2)/(abs(w2[1])+abs(w2[0]))
        w3=abs(w3)/(abs(w3[1])+abs(w3[0]))


        h1 = convmean(f,g, p = w[0] , q = w[1] )
        h2 = convmean(f,g, p = w2[0], q = w2[1] )
        h3 = convmean(f,g, p = w3[0], q = w3[1] )
        print("fun1:", fun1(w), fun1(w2), fun1(w3))
        print("fun2:", fun2(w), fun2(w2), fun2(w3))
        print("fun3:", fun3(w), fun3(w2), fun3(w3))
        print("w1=", w, h1.var(), h1.iqrange(0.01), h1.medianad())
        print("w2=", w2, h2.var(), h2.iqrange(0.01), h2.medianad())
        print("w3=", w3, h3.var(), h3.iqrange(0.01), h3.medianad())
        h1.plot(linewidth=2, color = 'b', linestyle='-')
        h2.plot(linewidth=2, color = 'k', linestyle='-')
        h3.plot(linewidth=2, color = 'm', linestyle='-')
        plt.figure()
        c1 = h1.cumint();
        c2 = h2.cumint();
        c3 = h3.cumint();
        print(f)
        print(f.summary())
        print(g)
        print(g.summary())
        print(h1)
        print(h1.summary())
        print(h2)
        print(h2.summary())
        print(h3)
        print(h3.summary())
        c1.plot(linewidth=1, color = 'b', linestyle='-')
        c2.plot(linewidth=1, color = 'k', linestyle='-')
        c3.plot(linewidth=1, color = 'm', linestyle='-')
        int = h1.integrate()
        self.assertTrue(abs(int-1)<self.tol, 'integral = {0}'.format(abs(int-1)))

    @unittest.expectedFailure
    def testConvmeanBeta(self):
        """Weighted mean of two uniform random variables
         with different variance, optimal solution"""
        plt.figure()
        segf1 = Segment(1.0, 2.0, f_helper2)#lambda x: 2772.0*betapdf(x-1,alpha = 6, beta=6))
        segf2 = ConstSegment(1.5, 4.5, 1.0/3.0)
        f = PiecewiseFunction([])
        f.addSegment(segf1)
        g = PiecewiseFunction([])
        g.addSegment(segf2)
        segn1 = MInfSegment(-0.0, normpdf_1)
        segn2 = PInfSegment(2.0,  normpdf_1)
        segn3 = Segment(0.0, 2.0, normpdf_1)
        n = PiecewiseFunction([])
        n.addSegment(segn1)
        n.addSegment(segn2)
        n.addSegment(segn3)
        n.plot(color='r')
        g.plot(color='g')
        #mf = f.mean()
        #vf = f.var()
        #mg = g.mean()
        #vg = g.var()
        h = convmean(n,g, p=0.57142393, q= 0.42857607)
        h.plot(linewidth=1, color = 'k', linestyle='-')
        int = h.integrate()
        #print(n.var(), g.var(), h.var())
        #w0=[0.5, 0.5]
        #fun = lambda w : convmean(n,g, p=w[0],q=w[1]).var();
        #w = fmin(fun, w0)
        #w=w/(w[1]+w[0])
        #print w
        #print fun(w)
    def testThChi2(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 3, op = conv, theoretic = chisqr, a = 0, b=Inf, isPoleAtZero = True)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testThNormal(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 3, op = conv, theoretic = f_helper3, a = -Inf, b=Inf, isPoleAtZero = False)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testThCauchy(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 3, op = conv, theoretic = cauchy, a = -Inf, b=Inf, isPoleAtZero = False, plot_tails = True, asympf = f_m2x)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testThLevy05(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 2, op = conv, theoretic = f_helper4, a = 0, b=Inf, isPoleAtZero = False, splitPoints = array([1.0/3]), plot_tails = True, asympf = lambda x: -1.5*x)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    @unittest.skip("loss of accuracy")
    def testOneOverSqrRoot(self):
        plt.figure()
        f = PiecewiseFunction([])
        f.addSegment(SegmentWithPole(0, 1, sqrrrootdistr))
        ints = do_testWithTheoretical(n = 2, op = conv, f = f, theoretic = sqrrrootdistr, comp_w_theor = False, a = 0, b=1, isPoleAtZero = True)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    @unittest.skip("loss of accuracy")
    def testOneOverSqrRootAbs(self):
        plt.figure()
        f = PiecewiseFunction([])
        f.addSegment(SegmentWithPole(-1, 0, sqrrrootdistr2, left_pole = False))
        f.addSegment(SegmentWithPole(0, 1, sqrrrootdistr2))
        ints = do_testWithTheoretical(n = 2, op = conv, f = f, theoretic = sqrrrootdistr2, comp_w_theor = False, a = -1, b=1, isPoleAtZero = True)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testThNegChi2(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 2, op = conv, theoretic = neg_chisq, a = -Inf, b=0, isPoleAtZero = True)
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testNegLog(self):
        f = PiecewiseFunction([])
        f.addSegment(SegmentWithPole(0, 0.5, logdistr))
        f.addSegment(Segment(0.5, 1, logdistr))
        plt.figure()
        ints = do_testWithTheoretical(n = 3, op = conv, f = f, theoretic = logdistr, comp_w_theor = False, a = 0, b=1, isPoleAtZero = True, splitPoints = [0.5])
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))
    def testNegLogAbs(self):
        f = PiecewiseFunction([])
        f.addSegment(Segment(-1, -0.5, logdistr2))
        f.addSegment(SegmentWithPole(-0.5, 0, logdistr2, left_pole = False))
        f.addSegment(SegmentWithPole(0, 0.5, logdistr2))
        f.addSegment(Segment(0.5, 1, logdistr2))
        plt.figure()
        ints = do_testWithTheoretical(n = 3, op = conv, f = f, theoretic = logdistr, comp_w_theor = False, a = 0, b=1, isPoleAtZero = True, splitPoints = [0.5])
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))


    def testThProdUni(self):
        plt.figure()
        ints = do_testWithTheoretical(n = 2, op = convprod, theoretic = prodNuniform01, a = 0, b = 1, isPoleAtZero = False, splitPoints = array([0.5]))
        self.assertTrue(max(abs(ints-1.0))<self.tol, 'integral error = {0}'.format(max(abs(ints-1))))

    def testRatio2(self):
        plt.figure()
        fun = PiecewiseFunction([])
        segf1 = Segment(0.0, 1.0, operator.pos)
        segf2 = Segment(1.0, 2.0, f_compl)
        fun.addSegment(segf1)
        fun.addSegment(segf2)
        err = ratioTester(fun)
        self.assertTrue(err < self.tol, 'integral error = {0}'.format(err))
    def testRatio3(self):
        plt.figure()
        fun = PiecewiseFunction([])
        segf1 = Segment(0.0, 1.0, f_one)
        fun.addSegment(segf1)
        err = ratioTester(fun)
        self.assertTrue(err < self.tol, 'integral error = {0}'.format(err))

    def testInversionUniform(self):
        plt.figure()
        fun = PiecewiseFunction([])
        #segf1 = Segment(0.0, 1.0, lambda x: x)
        segf2 = Segment(0.5, 2.0, f_2_3)
        #fun.addSegment(segf1)
        fun.addSegment(segf2)
        err = ratioTester(fun)
        self.assertTrue(err < self.tol, 'integral error = {0}'.format(err))
    def testInversionNormal(self):
        plt.figure()
        fun = PiecewiseFunction([])
        segf1 = MInfSegment(-1.0, normpdf)
        segf2 = PInfSegment(1.0,  normpdf)
        segf3 = Segment(-1.0, 0,  normpdf)
        segf4 = Segment(0, 1.0,   normpdf)
        fun.addSegment(segf1)
        fun.addSegment(segf2)
        fun.addSegment(segf3)
        fun.addSegment(segf4)
        err = ratioTester(fun)
        self.assertTrue(err < self.tol, 'integral error = {0}'.format(err))
    def testIntegration(self):
        segf1 = Segment(1.0, 2.0, f_helper2)#lambda x: 2772.0*betapdf(x-1,alpha = 6, beta=6))
        segf2 = ConstSegment(0.0, 1.0, 0.5/5.0)
        segf3 = ConstSegment(1.0, 3.0, 1.0/5.0)
        segf4 = Segment(3.0, 4.0, f_4mx)

        segg1 = MInfSegment(-1.0, normpdf)
        segg2 = PInfSegment(1.0,  normpdf)
        segg3 = Segment(-1.0, 0,  normpdf)
        segg4 = Segment(0, 1.0,   normpdf)

        segh1 = SegmentWithPole(0.0, 1.0, chisqr)
        segh2 = PInfSegment(1.0, chisqr)

        segk1 = SegmentWithPole( 0.0, 0.1, m_half_log_abs)
        segk2 = SegmentWithPole(-0.1, 0.0, m_half_log_abs, left_pole= False)
        segk3 = Segment(-1.0, -0.1, m_half_log_abs)
        segk4 = Segment( 0.1,  1.0, m_half_log_abs)

        #segk1 = SegmentWithPole( 0.0, 0.1, lambda x:0.25/(abs(x))**0.5)
        #segk2 = SegmentWithPole(-0.1, 0.0, lambda x:0.25/(abs(x))**0.5, left_pole= False)
        #segk3 = Segment(-1.0, -0.1, lambda x:0.25/(abs(x))**0.5)
        #segk4 = Segment( 0.1,  1.0, lambda x:0.25/(abs(x))**0.5)

        f1 = PiecewiseFunction([])
        f2 = PiecewiseFunction([])
        g3 = PiecewiseFunction([])
        g4 = PiecewiseFunction([])
        g5 = PiecewiseFunction([])

        f1.addSegment(segf1);

        f2.addSegment(segf2);
        f2.addSegment(segf3);
        f2.addSegment(segf4);

        g3.addSegment(segg1);
        g3.addSegment(segg2);
        g3.addSegment(segg3);
        g3.addSegment(segg4);

        g4.addSegment(segh1);
        g4.addSegment(segh2);

        g5.addSegment(segk1);
        g5.addSegment(segk2);
        g5.addSegment(segk3);
        g5.addSegment(segk4);
        for fi in [f1, f2, g3, g4, g5]:
            plt.figure()
            print("==========================================================")
            #print(fi.summary());
            integrationTester(fi)

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(TestPicewiseConvs("testPiecewiseFunction"))

    #suite.addTest(TestPicewiseConvs("testConvUniform"))
    #suite.addTest(TestPicewiseConvs("testConvUniformMix"))
    #suite.addTest(TestPicewiseConvs("testConvCauchy"))
    #suite.addTest(TestPicewiseConvs("testConvXalpha"))
    #suite.addTest(TestPicewiseConvs("testConvNormalScaled"))
    #suite.addTest(TestPicewiseConvs("testConvStable0_5"))

    #suite.addTest(TestPicewiseConvs("testConvprodUniform"))
    #suite.addTest(TestPicewiseConvs("testConvprodMix"))
    #suite.addTest(TestPicewiseConvs("testConvprodUniform2"))
    #suite.addTest(TestPicewiseConvs("testConvprodUniform01"))

    ##suite.addTest(TestPicewiseConvs("testConvprodCauchy"))
    #suite.addTest(TestPicewiseConvs("testConvprodCauchyUni"))
    #suite.addTest(TestPicewiseConvs("testConvprodMix2"))
    #suite.addTest(TestPicewiseConvs("testConvProdNormal"))
    #suite.addTest(TestPicewiseConvs("testConvProdSqrtUni"))
    #suite.addTest(TestPicewiseConvs("testConvProdInf1"))

    #suite.addTest(TestPicewiseConvs("testConvDivInf1"))
    #suite.addTest(TestPicewiseConvs("testConvDivNormal"))
    #suite.addTest(TestPicewiseConvs("testConvDivCauchy"))
    #suite.addTest(TestPicewiseConvs("testCauchy"))
    #suite.addTest(TestPicewiseConvs("testConvDivUni"))
    #suite.addTest(TestPicewiseConvs("testConvDivUni2"))

    #suite.addTest(TestPicewiseConvs("testConvDivUniWide"))
    #suite.addTest(TestPicewiseConvs("testConvDivMix"))


    #suite.addTest(TestPicewiseConvs("testConvMax"))
    #suite.addTest(TestPicewiseConvs("testConvmaxNormal"))
    #suite.addTest(TestPicewiseConvs("testConvMin"))

    #suite.addTest(TestPicewiseConvs("testSegmentsWithPoles"))

    #suite.addTest(TestPicewiseConvs("testSquareNormal"))

    #suite.addTest(TestPicewiseConvs("testLogNotOneOverX"))

    #suite.addTest(TestPicewiseConvs("testRatio"))

    #suite.addTest(TestPicewiseConvs("testConvDiscrete"))
    #suite.addTest(TestPicewiseConvs("testConvDiscreteMix1"))
    suite.addTest(TestPicewiseConvs("testConvDiscreteMix2"))
    #suite.addTest(TestPicewiseConvs("testConvmeanUniform"))
    #suite.addTest(TestPicewiseConvs("testConvmeanBeta"))

    #suite.addTest(TestPicewiseConvs("testThNormal"))
    #suite.addTest(TestPicewiseConvs("testThCauchy"))
    #suite.addTest(TestPicewiseConvs("testThLevy05"))
    #suite.addTest(TestPicewiseConvs("testThChi2"))
    #suite.addTest(TestPicewiseConvs("testOneOverSqrRoot"))
    #suite.addTest(TestPicewiseConvs("testOneOverSqrRootAbs"))
    #suite.addTest(TestPicewiseConvs("testThNegChi2"))
    #suite.addTest(TestPicewiseConvs("testNegLog"))
    #suite.addTest(TestPicewiseConvs("testNegLogAbs"))


    #suite.addTest(TestPicewiseConvs("testThProdUni"))

    #suite.addTest(TestPicewiseConvs("testRatio2"))
    #suite.addTest(TestPicewiseConvs("testRatio3"))

    #suite.addTest(TestPicewiseConvs("testInversionUniform"))
    #suite.addTest(TestPicewiseConvs("testInversionNormal"))

    #suite.addTest(TestPicewiseConvs("testIntegration"))

    return suite;

if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    import warnings
    warnings.filterwarnings('always')

    runner = unittest.TextTestRunner()
    test = suite()
    runner.run(test)
    show()

    #unittest.main()
