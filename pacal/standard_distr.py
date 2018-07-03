"""Standard distributions."""

from __future__ import print_function

import numpy as np
from numpy import isscalar, zeros_like, asfarray, zeros, cumsum, array, searchsorted, bincount
from numpy import pi, sqrt, exp, log, log1p, cos, floor
from numpy.random import normal, uniform, chisquare, exponential, gamma, beta, pareto, laplace, standard_t, weibull, gumbel, randint
from numpy.random import f as f_rand

from numpy import finfo, double
from functools import partial
_MAX_EXP_ARG = log(finfo(double).max)

from . import params
from .utils import lgamma, wrap_pdf
from .distr import Distr, DiscreteDistr, ConstDistr
from .segments import PiecewiseFunction, PiecewiseDistribution, Segment
from .segments import ConstSegment, PInfSegment, MInfSegment, SegmentWithPole

try:
    from numpy import Inf
except:
    Inf = float('inf')

class FunDistr(Distr):
    """General distribution defined as function with
    singularities at given breakPoints."""
    def __init__(self, fun, breakPoints = None, interpolated = False, **kwargs):
        super(FunDistr, self).__init__(**kwargs)
        self.fun = fun
        self.breakPoints = breakPoints
        if "sym" in kwargs:
            kwargs.pop("sym")
        self.kwargs = kwargs
        self.interpolated = interpolated
    def pdf(self, x):
        return self.fun(x)
    def getName(self):
        return "FUN({0},{1})".format(self.breakPoints[0], self.breakPoints[-1])
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution(fun = self.fun, breakPoints = self.breakPoints, **self.kwargs)
        if self.interpolated:
            self.piecewise_pdf = self.piecewise_pdf.toInterpolated()
    def rand_raw(self, n = 1):
        return self.rand_invcdf(n)
    def range(self):
        return self.breakPoints[0], self.breakPoints[-1]

class PDistr(Distr):
    """General distribution defined as piecewise function."""
    def __init__(self, segs = None, **kvargs):
        super(PDistr, self).__init__(**kvargs)
        self.segs = segs
    def init_piecewise_pdf(self):
        if isinstance(self.segs, PiecewiseFunction):
            self.piecewise_pdf = self.segs
        else:
            self.piecewise_pdf = PiecewiseDistribution();
            for seg in self.segs:
                self.piecewise_pdf.addSegment(seg)
    def getName(self):
        #return "PDISTR({0})".format(self.get_piecewise_pdf())
        return "PDISTR({0},{1})".format(self.get_piecewise_pdf().segments[0].a, self.get_piecewise_pdf().segments[-1].b)
    def rand_raw(self, n = 1):
        return self.rand_invcdf(n)

class MixDistr(Distr):
    """Mixture of distributions"""
    def __init__(self, probs, distrs):
        """Keyword arguments:
        probs -- list of pi's
        distrs -- list of distributions
        """
        super(MixDistr, self).__init__()
        assert len(probs) == len(distrs)
        self.nmix = len(probs)
        self.probs = probs
        self.distrs = distrs
    def init_piecewise_pdf(self):
#        mixdistr = ConstDistr(1, self.probs[0]) * self.distrs[0]
#        for i in range(1,len(self.probs)):
#            mixi = ConstDistr(1,self.probs[i]) * self.distrs[i]
#            print self.probs[i], self.distrs[i]
#            mixdistr.piecewise_pdf = mixdistr.get_piecewise_pdf() + mixi.get_piecewise_pdf()
#        self.piecewise_pdf = mixdistr.piecewise_pdf
        mix_pdf =  self.probs[0] * self.distrs[0].get_piecewise_pdf()
        for i in range(1,len(self.probs)):
            #mixi = ConstDistr(1,self.probs[i]) * self.distrs[i]
            #print self.probs[i], self.distrs[i]
            mix_pdf += self.probs[i] * self.distrs[i].get_piecewise_pdf()
        self.piecewise_pdf = mix_pdf
        #self.piecewise_pdf = PiecewiseDistribution(fun=mixdistr.piecewise_pdf, breakPoints = mixdistr.piecewise_pdf.getBreaks())
    def getName(self):
        return "MIX()".format()
    def rand_raw(self, n = 1):
        r = zeros(n)
        idx = randint(0, self.nmix, n)
        nk = bincount(idx, minlength = self.nmix)
        for k in range(self.nmix):
            r[idx == k] = self.distrs[k].rand(nk[k])
        return r
    def range(self):
        rs = [d.range() for d in self.distrs]
        a = min(r[0] for r in rs)
        b = max(r[1] for r in rs)
        return a, b
    #def pdf(self, x):
    #    return self.fun(x)
    #def init_piecewise_pdf(self):
    #    self.piecewise_pdf = PiecewiseDistribution(fun = self.fun, breakPoints = self.breakPoints)
class ExtremeMixDistr(Distr):
    """Mixture of distributions"""
    def __init__(self, probs, distrs, cuts):
        """Keyword arguments:
        probs -- list of pi's
        distrs -- list of distributions
        """
        super(MixDistr, self).__init__()
        assert len(probs) == len(distrs)
        self.nmix = len(probs)
        self.probs = probs
        self.distrs = distrs
        self.cuts = cuts
    def init_piecewise_pdf(self):
        mix_pdf =  self.probs[0] * self.distrs[0].get_piecewise_pdf()
        for i in range(1,len(self.probs)):
            #mixi = ConstDistr(1,self.probs[i]) * self.distrs[i]
            #print self.probs[i], self.distrs[i]
            mix_pdf += self.probs[i] * self.distrs[i].get_piecewise_pdf()
        self.piecewise_pdf = mix_pdf
        #self.piecewise_pdf = PiecewiseDistribution(fun=mixdistr.piecewise_pdf, breakPoints = mixdistr.piecewise_pdf.getBreaks())
    def getName(self):
        return "MIX()".format()

    #def pdf(self, x):
    #    return self.fun(x)
    #def init_piecewise_pdf(self):
    #    self.piecewise_pdf = PiecewiseDistribution(fun = self.fun, breakPoints = self.breakPoints)

class NormalDistr(Distr):
    def __init__(self, mu=0.0, sigma=1.0, **kwargs):
        super(NormalDistr, self).__init__(**kwargs)
        if sigma <= 0:
            raise ValueError("Standard deviation of normal distribution must be nonnegative")
        self.mu = mu
        self.sigma = sigma
        self.one_over_twosigma2 = 0.5 / (sigma * sigma)
        self.nrm = 1.0 / (self.sigma * sqrt(2*pi))
    def init_piecewise_pdf(self):
        # put breaks at inflection points
        b1 = self.mu - self.sigma
        b2 = self.mu + self.sigma
        #pdf = partial(_norm_pdf, self.mu, self.nrm, self.one_over_twosigma2)
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(MInfSegment(b1, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(b1, b2, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(b2, wrapped_pdf))
    def pdf(self, x):
        q = (x-self.mu)**2 * self.one_over_twosigma2
        f = self.nrm * exp(-q)
        return f
    def rand_raw(self, n = None):  # None means return scalar
        return normal(self.mu, self.sigma, n)
    def __str__(self):
        return "Normal({0},{1})#{2}".format(self.mu, self.sigma, self.id())
    def getName(self):
        return "N({0},{1})".format(self.mu, self.sigma)
    def range(self):
        return -Inf, Inf

class UniformDistr(Distr):
    def __init__(self, a = 0.0, b = 1.0, **kwargs):
        super(UniformDistr, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.p = 1.0 / float(b-a)
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        self.piecewise_pdf.addSegment(ConstSegment(self.a, self.b, self.p))
    def rand_raw(self, n = None):
        return uniform(self.a, self.b, n)
    def __str__(self):
        return "Uniform({0},{1})#{2}".format(self.a, self.b, self.id())
    def getName(self):
        return "U({0},{1})".format(self.a, self.b)
    def range(self):
        return self.a, self.b

def _lin_fun1(a, b, u, x):
    return u * (x - a) / (b - a)
def _lin_fun2(c, d, u, x):
    return u * (d - x) / (d - c)
class TrapezoidalDistr(Distr):
    def __init__(self, a=0.0, b=0.0, c=1.0, d=1.0, **kwargs):
        super(TrapezoidalDistr, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.u = 2.0 / (d + c - (b + a))
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        if self.a<self.b:
            self.piecewise_pdf.addSegment(Segment(self.a, self.b, partial(_lin_fun1, self.a, self.b, self.u)))
        if self.b<self.c:
            self.piecewise_pdf.addSegment(ConstSegment(self.b, self.c, self.u))
        if self.c<self.d:
            self.piecewise_pdf.addSegment(Segment(self.c, self.d, partial(_lin_fun2, self.c, self.d, self.u)))
    def rand_raw(self, n=None):
        return  self.rand_invcdf(n) # TODO !to improve it!
    def __str__(self):
        return "Trapz({0},{1},{2},{3})#{4}".format(self.a, self.b,self.c, self.d, self.id())
    def getName(self):
        return "Trapz({0},{1},{2},{3})".format(self.a, self.b,self.c, self.d)
    def range(self):
        return self.a, self.d

class CauchyDistr(Distr):
    def __init__(self, gamma = 1.0, center = 0.0, **kwargs):
        super(CauchyDistr, self).__init__(**kwargs)
        self.gamma = gamma
        self.center = center
    def pdf(self, x):
        return self.gamma / (pi * (self.gamma*self.gamma + (x - self.center)*(x - self.center)))
    def init_piecewise_pdf(self):
        b1 = self.center - self.gamma
        b2 = self.center + self.gamma
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(MInfSegment(b1, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(b1, b2, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(b2, wrapped_pdf))
        #self.piecewise_pdf = self.piecewise_pdf.toInterpolated()
    def rand_raw(self, n = None):
        return self.center + normal(0, 1, n) / normal(0, 1, n) * self.gamma
    def __str__(self):
        if self.gamma == 1 and self.center == 0:
            return "Cauchy#{0}".format(self.id())
        else:
            return "Cauchy(gamma={0}, center={1})#{2}".format(self.gamma, self.center, self.id())
    def getName(self):
        return "Cauchy({0},{1})".format(self.center, self.gamma)
    def range(self):
        return -Inf, Inf

class ChiSquareDistr(Distr):
    def __init__(self, df = 1, **kwargs):
        assert df > 0
        super(ChiSquareDistr, self).__init__(**kwargs)
        self.df = df
        self.df2 = df / 2.0
        self.lg_norm = lgamma(self.df2) + self.df2 * log(2)
        if self.df == 1:
            self.pdf_at_0 = Inf
            self.lpdf_at_0 = Inf
        elif self.df == 2:
            self.pdf_at_0 = 0.5
            self.lpdf_at_0 = log(0.5)
        else:
            self.pdf_at_0 = 0
            self.lpdf_at_0 = -Inf
    def log_pdf(self, x):
        lpdf = (self.df2 - 1) * log(x) - x/2.0 - self.lg_norm
        return lpdf
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            elif x == 0:
                y = self.pdf_at_0
            else:
                y = exp(self.log_pdf(x))
        else:
            y = zeros_like(asfarray(x))
            mask = (x > 0)
            y[mask] = exp(self.log_pdf(x[mask]))
            mask_zero = (x == 0)
            y[mask_zero] = self.pdf_at_0
        return y
    def init_piecewise_pdf(self):
        wrapped_pdf = wrap_pdf(self.pdf)
        if 1 <= self.df <= 20:
            self.piecewise_pdf = PiecewiseDistribution(fun = wrapped_pdf,
                                                   breakPoints = [0.0, self.df/2.0, self.df*2.0, Inf],
                                                   lpoles=[True, False, False, False])
        elif 20 < self.df:
            mean = self.df
            std = sqrt(2 * self.df)
            self.piecewise_pdf = PiecewiseDistribution(fun = wrapped_pdf,
                                                        breakPoints =[0.0, self.df*0.75, self.df*4.0/3.0,  Inf],
                                                        lpoles=[True, False, False, False])

        else:
            print("unexeepted df=", self.df)
        #if self.df == 1 or self.df == 3:
        #    self.piecewise_pdf.addSegment(SegmentWithPole(0, 1, self.pdf, left_pole = True))
        #else:
        #    self.piecewise_pdf.addSegment(Segment(0, 1, self.pdf))
        #if self.df <= 3:
        #    self.piecewise_pdf.addSegment(PInfSegment(1, self.pdf))
        #elif self.df <= 6:
        #    mode = self.df - 2
        #    self.piecewise_pdf.addSegment(Segment(1, mode, self.pdf))
        #    self.piecewise_pdf.addSegment(Segment(mode, 2*mode, self.pdf))
        #    self.piecewise_pdf.addSegment(PInfSegment(2*mode, self.pdf))
        #else:
        #    mean = self.df
        #    std = sqrt(2 * self.df)
        #    self.piecewise_pdf.addSegment(Segment(1, mean - std, self.pdf))
        #    self.piecewise_pdf.addSegment(Segment(mean - std, mean + std, self.pdf))
        #    self.piecewise_pdf.addSegment(PInfSegment(mean + std, self.pdf))
    def rand_raw(self, n = None):
        return chisquare(self.df, n)
    def __str__(self):
        return "ChiSquare(df={0})#{1}".format(self.df, self.id())
    def getName(self):
        return "Chi2({0})".format(self.df)
    def range(self):
        return 0.0, Inf

class ExponentialDistr(Distr):
    def __init__(self, lmbda = 1, **kwargs):
        super(ExponentialDistr, self).__init__(**kwargs)
        self.lmbda = lmbda
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            else:
                y = self.lmbda * exp(-self.lmbda * x)
        else:
            y = zeros_like(asfarray(x))
            mask = (x >= 0)
            y[mask] = self.lmbda * exp(-self.lmbda * x[mask])
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(0, 1, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(1, wrapped_pdf))
    def rand_raw(self, n = None):
        return exponential(1.0/self.lmbda, n)
    def __str__(self):
        return "Exponential(lambda={0})#{1}".format(self.lmbda, self.id())
    def getName(self):
        return "ExpD({0})".format(self.lmbda)
    def range(self):
        return 0.0, Inf

class GammaDistr(Distr):
    def __init__(self, k = 2, theta = 2, **kwargs):
        super(GammaDistr, self).__init__(**kwargs)
        assert k > 0
        assert theta > 0
        self.k = k
        self.theta = theta
        self.lg_norm = lgamma(self.k) + self.k * log(self.theta)
        if self.k < 1:
            self.pdf_at_0 = Inf
            self.lpdf_at_0 = Inf
        elif self.k == 1:
            self.pdf_at_0 = 1.0 / self.theta
            self.lpdf_at_0 = -log(self.theta)
        else:
            self.pdf_at_0 = 0
            self.lpdf_at_0 = -Inf
    def log_pdf(self, x):
        lpdf = (self.k - 1) * log(x) - x / self.theta - self.lg_norm
        return lpdf
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            elif x == 0:
                y = self.pdf_at_0
            else:
                y = exp(self.log_pdf(x))
        else:
            y = zeros_like(asfarray(x))
            mask = (x > 0)
            y[mask] = exp(self.log_pdf(x[mask]))
            mask_zero = (x == 0)
            y[mask_zero] = self.pdf_at_0
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        if self.k < 1:
            self.piecewise_pdf.addSegment(SegmentWithPole(0, 1, wrapped_pdf, left_pole = True))
            self.piecewise_pdf.addSegment(PInfSegment(1, wrapped_pdf))
        elif self.k == 1:
            self.piecewise_pdf.addSegment(Segment(0, 1, wrapped_pdf))
            self.piecewise_pdf.addSegment(PInfSegment(1, wrapped_pdf))
        else:
            mode = (self.k - 1) * self.theta
            self.piecewise_pdf.addSegment(Segment(0, mode / 2, wrapped_pdf))
            self.piecewise_pdf.addSegment(Segment(mode / 2, mode, wrapped_pdf))
            self.piecewise_pdf.addSegment(Segment(mode, 2 * mode, wrapped_pdf))
            self.piecewise_pdf.addSegment(PInfSegment(2*mode, wrapped_pdf))
    def rand_raw(self, n = None):
        return gamma(self.k, self.theta, n)
    def __str__(self):
        return "Gamma(k={0},theta={1})#{2}".format(self.k, self.theta, self.id())
    def getName(self):
        return "Gamma({0},{1})".format(self.k, self.theta)
    def range(self):
        return 0.0, Inf

class BetaDistr(Distr):
    def __init__(self, alpha = 1, beta = 1, **kwargs):
        super(BetaDistr, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.lg_norm = lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta)
    def pdf(self, x):
        if isscalar(x):
            if x < 0 or x > 1:
                y = 0
            else:
                y = exp(self.lg_norm + log(x)*(self.alpha - 1) + log(1-x)*(self.beta - 1))
        else:
            y = zeros_like(asfarray(x))
            mask = (x >= 0) & (x <= 1)
            y[mask] = exp(self.lg_norm + log(x[mask])*(self.alpha - 1) + log(1-x[mask])*(self.beta - 1))
        return y
    def init_piecewise_pdf(self):
        if self.alpha > 1 and self.beta > 1:
            m = float(self.alpha - 1) / (self.alpha + self.beta - 2) # mode
        else:
            m = 0.5
        m = 0.5 # TODO check this, but it seems better
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        poleL = self.alpha < 2 and abs(self.alpha - 1) > params.pole_detection.max_pole_exponent
        poleR = self.beta < 2  and abs(self.beta - 1) > params.pole_detection.max_pole_exponent
        if poleL and poleR:
            self.piecewise_pdf.addSegment(SegmentWithPole(0, m, wrapped_pdf, left_pole = True))
            self.piecewise_pdf.addSegment(SegmentWithPole(m, 1, wrapped_pdf, left_pole = False))
        elif poleL:
            self.piecewise_pdf.addSegment(SegmentWithPole(0, m, wrapped_pdf, left_pole = True))
            self.piecewise_pdf.addSegment(Segment(m, 1, wrapped_pdf))
        elif poleR:
            self.piecewise_pdf.addSegment(Segment(0, m, wrapped_pdf))
            self.piecewise_pdf.addSegment(SegmentWithPole(m, 1, wrapped_pdf, left_pole = False))
        else:
            self.piecewise_pdf.addSegment(Segment(0, m, wrapped_pdf))
            self.piecewise_pdf.addSegment(Segment(m, 1, wrapped_pdf))
    def rand_raw(self, n = None):
        return beta(self.alpha, self.beta, n)
    def __str__(self):
        return "Beta(alpha={0},beta={1})#{2}".format(self.alpha, self.beta, self.id())
    def getName(self):
        return "Beta({0},{1})".format(self.alpha, self.beta)
    def range(self):
        return 0.0, 1.0

class ParetoDistr(Distr):
    def __init__(self, alpha = 1, xmin = 1, **kwargs):
        assert alpha > 0
        assert xmin > 0
        super(ParetoDistr, self).__init__(**kwargs)
        self.alpha = alpha
        self.xmin = xmin
        self.nrm = float(alpha * xmin ** alpha)
    def pdf(self, x):
        if isscalar(x):
            if x < self.xmin:
                y = 0
            else:
                y = self.nrm / x ** (self.alpha + 1)
        else:
            y = zeros_like(asfarray(x))
            mask = (x >= self.xmin)
            y[mask] = self.nrm / x[mask] ** (self.alpha + 1)
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(self.xmin, self.xmin + 1, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(self.xmin + 1, wrapped_pdf))
    def rand_raw(self, n = None):
        return self.xmin + pareto(self.alpha, n) * self.xmin
    def __str__(self):
        return "ParetoDistr(alpha={0},xmin={1})#{2}".format(self.alpha, self.xmin, self.id())
    def getName(self):
        return "Pareto({0},{1})".format(self.alpha, self.xmin)
    def range(self):
        return self.xmin, Inf

class LevyDistr(Distr):
    def __init__(self, c = 1.0, xmin = 0.0, **kwargs):
        assert c > 0
        super(LevyDistr, self).__init__(**kwargs)
        self.c = c
        self.xmin = xmin
        self.nrm = sqrt(c/(2*pi))
    def pdf(self, x):
        if isscalar(x):
            if x <= self.xmin:
                y = 0
            else:
                y = self.nrm * exp(log(x - self.xmin)*(-1.5) -0.5 * self.c / (x - self.xmin))
        else:
            y = zeros_like(asfarray(x))
            mask = (x > self.xmin)
            y[mask] = self.nrm * exp(log(x[mask] - self.xmin)*(-1.5) -0.5 * self.c / (x[mask] - self.xmin))
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(self.xmin, self.xmin + self.c, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(self.xmin + self.c, wrapped_pdf))
    def rand_raw(self, n = None):
        sigma = 1.0 / sqrt(self.c)
        return self.xmin + 1.0 / normal(0, sigma, n) ** 2
    def __str__(self):
        return "LevyDistr(c={0},xmin={1})#{2}".format(self.c, self.xmin, self.id())
    def getName(self):
        return "Levy({0},{1})".format(self.c, self.xmin)
    def range(self):
        return self.xmin, Inf

class LaplaceDistr(Distr):
    def __init__(self, lmbda = 1.0, mu = 0.0, **kwargs):
        assert lmbda > 0
        super(LaplaceDistr, self).__init__(**kwargs)
        self.lmbda = lmbda
        self.mu = mu
        self.nrm = 0.5 / self.lmbda
    def pdf(self, x):
        y = self.nrm * exp(-abs(x - self.mu)/self.lmbda)
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(MInfSegment(self.mu - 2 * self.lmbda, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(self.mu - 2 * self.lmbda, self.mu, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(self.mu, self.mu + 2 * self.lmbda, wrapped_pdf))
        #self.piecewise_pdf.addSegment(Segment(self.mu - 2 * self.lmbda, self.mu + 2 * self.lmbda, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(self.mu + 2 * self.lmbda, wrapped_pdf))
    def rand_raw(self, n = None):
        return laplace(self.mu, self.lmbda, n)
    def __str__(self):
        return "LaplaceDistr(lambda={0},mu={1})#{2}".format(self.lmbda, self.mu, self.id())
    def getName(self):
        return "Laplace({0},{1})".format(self.lmbda, self.mu)
    def range(self):
        return -Inf, Inf

class StudentTDistr(Distr):
    def __init__(self, df = 2, **kwargs):
        assert df > 0
        super(StudentTDistr, self).__init__(**kwargs)
        self.df = df
        self.lg_norm = lgamma(float(self.df + 1) / 2) - lgamma(float(self.df) / 2) - 0.5 * (log(self.df) + log(pi))
    def pdf(self, x):
        lgy = self.lg_norm - (float(self.df + 1) / 2) * log1p(x**2 / self.df)
        return exp(lgy)
    def init_piecewise_pdf(self):
        # split at inflection points
        infl = sqrt(float(self.df) / (self.df + 2))
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(MInfSegment(-infl, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(-infl, infl, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(infl, wrapped_pdf))
    def rand_raw(self, n = None):
        return standard_t(self.df, n)
    def __str__(self):
        return "StudentTDistr(df={0})#{1}".format(self.df, self.id())
    def getName(self):
        return "StudentT({0})".format(self.df)
    def range(self):
        return -Inf, Inf

class SemicircleDistr(Distr):
    def __init__(self, R = 1.0, **kwargs):
        assert R > 0
        super(SemicircleDistr, self).__init__(**kwargs)
        self.R = R
        self.norm = 2.0 / (pi * R * R)
    def pdf(self, x):
        if isscalar(x):
            if -self.R <= x <= self.R:
                y = self.norm * sqrt(self.R*self.R - x*x)
        else:
            mask = (-self.R <= x) & (x <= self.R)
            y = zeros_like(asfarray(x))
            y[mask] = self.norm * sqrt(self.R*self.R - x*x)
        return y
    def init_piecewise_pdf(self):
        # split at inflection points
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(SegmentWithPole(-self.R, -float(self.R) / 2, wrapped_pdf, left_pole = True))
        self.piecewise_pdf.addSegment(Segment(-float(self.R) / 2, float(self.R) / 2, wrapped_pdf))
        self.piecewise_pdf.addSegment(SegmentWithPole(float(self.R) / 2, self.R, wrapped_pdf, left_pole = False))
    def rand_raw(self, n = None):
        return self.R * sqrt(uniform(0, 1, n)) * cos(uniform(0, 1, n) * pi)
    def __str__(self):
        return "Semicircle(R={0})#{1}".format(self.R, self.id())
    def getName(self):
        return "Semicircle({0})".format(self.R)
    def range(self):
        return -self.R, self.R

class FDistr(Distr):
    def __init__(self, df1 = 1, df2 = 1, **kwargs):
        super(FDistr, self).__init__(**kwargs)
        self.df1 = float(df1)
        self.df2 = float(df2)
        self.lg_norm = self.df2 / 2 * log(self.df2) + lgamma((self.df1 + self.df2) / 2) - lgamma(self.df1 / 2) - lgamma(self.df2 / 2)
        if self.df1 < 2:
            self.pdf_at_0 = Inf
        elif self.df1 == 2:
            self.pdf_at_0 = 1
        else:
            self.pdf_at_0 = 0
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            elif x == 0:
                y = self.pdf_at_0
            else:
                lgy = self.lg_norm + 0.5 * (self.df1 * log(self.df1*x) - (self.df1 + self.df2) * log(self.df1 * x + self.df2)) - log(x)
                y = exp(lgy)
        else:
            y = zeros_like(asfarray(x))
            y[x==0] = self.pdf_at_0
            mask = (x > 0)
            lgy = self.lg_norm + 0.5 * (self.df1 * log(self.df1*x[mask]) - (self.df1 + self.df2) * log(self.df1 * x[mask] + self.df2)) - log(x[mask])
            y[mask] = exp(lgy)
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        if self.df1 < 2:
            self.piecewise_pdf.addSegment(SegmentWithPole(0, 1, wrapped_pdf, left_pole = True))
            self.piecewise_pdf.addSegment(PInfSegment(1, wrapped_pdf))
        elif self.df1 == 2:
            self.piecewise_pdf.addSegment(Segment(0, 1, wrapped_pdf))
            self.piecewise_pdf.addSegment(PInfSegment(1, wrapped_pdf))
        else:
            mode = float(self.df1 - 2) / self.df1 * float(self.df2) / (self.df2 + 2)
            self.piecewise_pdf.addSegment(SegmentWithPole(0, mode, wrapped_pdf, left_pole = True))
            self.piecewise_pdf.addSegment(Segment(mode, mode + 1, wrapped_pdf))
            self.piecewise_pdf.addSegment(PInfSegment(mode + 1, wrapped_pdf))
    def rand_raw(self, n = None):
        return f_rand(self.df1, self.df2, n)
    def __str__(self):
        return "F(df1={0},df2={1})#{2}".format(self.df1, self.df2, self.id())
    def getName(self):
        return "F({0},{1})".format(self.df1, self.df2)
    def range(self):
        return 0.0, Inf

class WeibullDistr(Distr):
    def __init__(self, k = 3, lmbda = 1, **kwargs):
        super(WeibullDistr, self).__init__(**kwargs)
        assert k > 0
        assert lmbda > 0
        self.k = k
        self.lmbda = lmbda
        self.nrm = float(self.k) / self.lmbda
        if self.k < 1:
            self.pdf_at_0 = Inf
        elif self.k == 1:
            self.pdf_at_0 = 1
        else:
            self.pdf_at_0 = 0
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            elif x == 0:
                y = self.pdf_at_0
            else:
                y = self.nrm * exp(log(x / self.lmbda)*(self.k-1) - (x / self.lmbda)**self.k)
        else:
            y = zeros_like(asfarray(x))
            mask = (x > 0)
            y[mask] = self.nrm * exp(log(x[mask] / self.lmbda)*(self.k-1) - (x[mask] / self.lmbda)**self.k)
            mask_zero = (x == 0)
            y[mask_zero] = self.pdf_at_0
        return y
    def init_piecewise_pdf(self):
        wrapped_pdf = wrap_pdf(self.pdf)
        if self.k <= 1:
            self.piecewise_pdf = PiecewiseDistribution(fun = wrapped_pdf,
                                                       breakPoints = [0.0, self.k, Inf],
                                                       lpoles=[True, False, False])
        else:
            mode = self.lmbda * (float(self.k - 1) / self.k)**(1.0/self.k)
            if self.k == floor(self.k):
                self.piecewise_pdf = PiecewiseDistribution(fun = wrapped_pdf,
                                                           breakPoints = [0.0, mode, Inf],
                                                           lpoles=[False, False, False])
            else:
                self.piecewise_pdf = PiecewiseDistribution(fun = wrapped_pdf,
                                                           breakPoints = [0.0, mode, Inf],
                                                           lpoles=[True, False, False])
    def rand_raw(self, n = None):
        return self.lmbda * weibull(self.k, n)
    def __str__(self):
        return "Weibull(k={0},lambda={1})#{2}".format(self.k, self.lmbda, self.id())
    def getName(self):
        return "Weibull({0},{1})".format(self.k, self.lmbda)
    def range(self):
        return 0.0, Inf

class GumbelDistr(Distr):
    def __init__(self, mu = 0, sigma = 1, **kwargs):
        assert sigma > 0
        super(GumbelDistr, self).__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.one_over_sigma = 1.0 / sigma
    def pdf(self, x):
        t = self.one_over_sigma * (self.mu - x)
        if isscalar(x):
            if t > _MAX_EXP_ARG:
                y = 0
            else:
                y = self.one_over_sigma * exp(t - exp(t))
        else:
            y = zeros_like(asfarray(x))
            mask = (t <= _MAX_EXP_ARG)
            y[mask] = self.one_over_sigma * exp(t[mask] - exp(t[mask]))
        return y
    def init_piecewise_pdf(self):
        # split at inflection points
        infl1 = self.mu - self.sigma * log((3 + sqrt(5))/2)
        infl2 = self.mu + self.sigma * log((3 + sqrt(5))/2)
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(MInfSegment(infl1, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(infl1, infl2, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(infl2, wrapped_pdf))
    def rand_raw(self, n = None):
        return gumbel(self.mu, self.sigma, n)
    def __str__(self):
        return "GumbelDistr(mu={0},sigma={1})#{2}".format(self.mu, self.sigma, self.id())
    def getName(self):
        return "Gumbel({0},{1})".format(self.mu, self.sigma)
    def range(self):
        return 0.0, Inf # TODO check it

class FrechetDistr(Distr):
    def __init__(self, alpha = 2, s = 1, m = 0, **kwargs):
        assert alpha > 0
        assert s > 0
        super(FrechetDistr, self).__init__(**kwargs)
        self.alpha = alpha
        self.s = float(s)
        self.m = m
        self.nrm = self.alpha / self.s
    def pdf(self, x):
        t = (x - self.m) / self.s
        if isscalar(x):
            if t <= 0:
                y = 0
            else:
                y = self.nrm * exp(-(self.alpha+1) * log(t) - t ** (-self.alpha))
        else:
            y = zeros_like(asfarray(x))
            mask = (t > 0)
            y[mask] = self.nrm * exp(-(self.alpha+1) * log(t[mask]) - t[mask] ** (-self.alpha))
        return y
    def init_piecewise_pdf(self):
        # split at inflection points
        a = self.alpha
        infl1 = a*(3*(a+1) - sqrt(1 + 6*a + 5*a**2))/(a+1)/(a+2)/2
        infl2 = a*(3*(a+1) + sqrt(1 + 6*a + 5*a**2))/(a+1)/(a+2)/2
        infl1 = self.m + (infl1 ** (1.0/a) * self.s)
        infl2 = self.m + (infl2 ** (1.0/a) * self.s)
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(self.m, infl1, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(infl1, infl2, wrapped_pdf))
        self.piecewise_pdf.addSegment(PInfSegment(infl2, wrapped_pdf))
    def rand_raw(self, n = None):
        x = uniform(0,1,n)
        return self.m + self.s*(-log(x))**(-1.0/self.alpha)
    def __str__(self):
        return "FrechetDistr(alpha={0},s={1},m={2})#{3}".format(self.alpha, self.s, self.m, self.id())
    def getName(self):
        return "Frechet({0},{1},{2})".format(self.alpha, self.s, self.m)
    def range(self):
        return 0.0, Inf # TODO check it

class MollifierDistr(Distr):
    """An infinitely smooth distribution which can be convolved with
    other distributions to smooth them out."""
    def __init__(self, epsilon = 1, **kwargs):
        assert epsilon > 0
        super(MollifierDistr, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.nrm = 2.252283621043581010499781255559830730074 / epsilon
    def pdf(self, x):
        if isscalar(x):
            if abs(x) >= self.epsilon:
                y = 0
            else:
                t = x / self.epsilon
                y = self.nrm * exp(-1.0 / (1 - t*t))
        else:
            y = zeros_like(asfarray(x))
            mask = abs(x) < self.epsilon
            y[mask] = self.nrm * exp(-1.0 / (1 - (x[mask] / self.epsilon)**2))
        return y
    def init_piecewise_pdf(self):
        # split at inflection points
        infl2 = 3.0 ** (-0.25) * self.epsilon
        infl1 = -infl2
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(-self.epsilon, infl1, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(infl1, infl2, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(infl2, self.epsilon, wrapped_pdf))
    def rand_raw(self, n = 1):
        return self.rand_invcdf(n)
    def __str__(self):
        return "MollifierDistr(epsilon={0})#{1}".format(self.epsilon, self.id())
    def getName(self):
        return "Mollifier({0})".format(self.epsilon)
    def range(self):
        return -self.epsilon, self.epsilon

### Discrete distributions

class ZeroDistr(ConstDistr):
    """One point distribution at point zero."""
    def __init__(self, **kwargs):
        super(ZeroDistr, self).__init__(c = 0.0, **kwargs)

class OneDistr(ConstDistr):
    """One point distribution at point one."""
    def __init__(self, **kwargs):
        super(OneDistr, self).__init__(c = 1.0, **kwargs)


class TruncDiscreteFunDistr(DiscreteDistr):
    """The truncated discrete distribution defined by sequence fun(k)."""
    def __init__(self, fun=None, trunk_eps=1e-16, **kwargs):
        self.trunk_eps = trunk_eps
        xi = []
        pi = []
        P = fun(0.0)
        k = 0.0
        S = 0.0
        while np.abs(S - 1.0) >= trunk_eps:
            xi.append(k)
            pi.append(fun(k))
            S = np.sum(pi)
            k += 1
        self.k_max = k
        pi = [p/S for p in pi]
        super(TruncDiscreteFunDistr, self).__init__(xi, pi, **kwargs)
    def __str__(self):
        return "TruncDiscreteFunDistr({0})#{1}".format(self.trunk_epsself.id())
    def getName(self):
        return "TruncDiscreteFunDistr({0})".format(self.trunk_eps)

class PoissonDistr(DiscreteDistr):
    """The truncated Poisson distribution."""
    def __init__(self, lmbda=1, trunk_eps=1e-15, **kwargs):
        self.lmbda = float(lmbda)
        self.trunk_eps = trunk_eps
        xi = []
        pi = []
        P = 1.0 * exp(-lmbda)
        k = 0.0
        S = P
        while (1.0-S) >= trunk_eps:
            xi.append(k)
            pi.append(P)
            k += 1.0
            P *= lmbda
            P /= k
            S += P
        xi.append(k)
        pi.append(P)
        self.k_max = len(xi)
        super(PoissonDistr, self).__init__(xi, pi, **kwargs)
    def __str__(self):
        return "Poisson({0})#{1}".format(self.lmbda, self.id())
    def getName(self):
        return "Poisson({0},{1})".format(self.lmbda, self.trunk_eps)

class BinomialDistr(DiscreteDistr):
    """The binomial distribution."""
    def __init__(self, n, p, **kwargs):
        self.n = n
        self.p = p
        self.q = 1-p
        P = self.q ** n
        xi = [0]
        pi = [P]
        pq = self.p / self.q
        for i in range(n):
            P *= n - i
            P /= i + 1
            P *= pq
            xi.append(i+1)
            pi.append(P)
        super(BinomialDistr, self).__init__(xi, pi, **kwargs)
    def __str__(self):
        return "Binomial({0}.{1})#{2}".format(self.n, self.p, self.id())
    def getName(self):
        return "Binom({0},{1})".format(self.n, self.p)

class BernoulliDistr(DiscreteDistr):
    def __init__(self, p=0.5, **kwargs):
        super(BernoulliDistr, self).__init__(xi=[0, 1], pi=[1.0-p, p], **kwargs)
    def __str__(self):
        return "Bernoulli({0})#{1}".format(self.pi[1], self.id())
    def getName(self):
        return "Bernoulli({0})".format(self.pi[1])


if __name__ == "__main__":
    from pylab import figure, show
    from .distr import demo_distr, log, exp, max, min, sqrt

    import numpy
    from numpy import ceil, log1p
    from scipy.special import gamma
    def poiss(k, lmbda=57.2):
        return lmbda**k * exp(-lmbda) /gamma(k+1.0)
    def powi(k, pow=2):
        return 1.0/(k+1.0)**(pow) * (6.0 / np.pi**2)

    B1 = BinomialDistr(n =10, p=0.2)
    B1.summary()
    P1 = PoissonDistr(lmbda=1.2)
    P1.summary()
    P2 = TruncDiscreteFunDistr(fun=powi, trunk_eps=1e-1)
    P2.summary()
    P1.plot()
    P2.plot()
    #print "c"
    #D  = OneDistr()/(P+0.1)
    #D.summary()
#    P.plot(color="k")
#    figure()
#    D.plot(color="r")
    P = P1 + B1
    figure()
#
    P.plot(color="r")
    P.get_piecewise_cdf().plot(color="r")
#
    P.summary()

    # M = MixDistr([0.5, 0.25, 0.125, 0.0625, 0.03125],
    #              [UniformDistr(-0.5,0.5)+4**0,
    #               UniformDistr(-0.5,0.5)+4**1,
    #               UniformDistr(-0.5,0.5)+4**2,
    #               UniformDistr(-0.5,0.5)+4**3,
    #               UniformDistr(-0.5,0.5)+4**4,
    #               ])
    # #M.plot()
    # (M/M).plot()
    # show()
    # 0/0
    #
    # mix = MixDistr([0.25, 0.75], [NormalDistr(-1,0.5), NormalDistr(1,2)])
    # print "======", mix.get_piecewise_pdf()
    # mix.summary()
    # mix.plot()
    # d = mix/mix
    # d.plot()
    # figure()
    # M = MixDistr([0.5, 0.25, 0.125, 0.0625, 0.03125],
    #              [UniformDistr(-1,1)/4+1,
    #              UniformDistr(-1,1)/8+2,
    #              UniformDistr(-1,1)/16+4,
    #              UniformDistr(-1,1)/32+8,
    #              UniformDistr(-1,1)/64+16])
    # M.summary()
    # M.plot()
    # M.get_piecewise_cdf().plot()
    # d= M/M
    # d.summary()
    # d.plot()



    #Ua = UniformDistr(1,2); #Ua.plot()
    #Ub = UniformDistr(0.25,2);
    #Uc = UniformDistr(0.25,2);
    #A = Ua + Ub
    #B = Ua - Ub
    #C = Ua * Ub
    #D = Ua / Ub
    #E = min(Ub, Uc)
    #print min(1,2)
    #F = max(Ub, Uc)
    #G = max(UniformDistr(0,3), UniformDistr(1,2))
    #H = min(UniformDistr(0,3), UniformDistr(1,2))
    #I = max(UniformDistr(0,0.1), UniformDistr(1,2))
    #J = min(UniformDistr(0,0.1), UniformDistr(1,2))
    #K = min(UniformDistr(0,1), NormalDistr(1,2))
    #demo_distr(A)
    #demo_distr(B)
    #demo_distr(C)
    #demo_distr(D)
    #demo_distr(E)
    #demo_distr(F)
    #demo_distr(G)
    #demo_distr(H)
    #demo_distr(I)
    #demo_distr(J)
    #demo_distr(K)

    # n = 4
    # T10 = (NormalDistr() / ChiSquareDistr(n)**0.5 ) * sqrt(n)
    # figure()
    # demo_distr(T10, theoretical = StudentTDistr(n), xmin = -1e10, xmax=1e10)
    #
    # N1 = NormalDistr()
    # num  = (N1 + N1 + N1 + N1+ N1) #/ 5**0.5
    # C1 = ChiSquareDistr(1)
    # #C1 = N1**2

    # den = (C1 + C1 + C1 + C1 + C1)
    # T5 = num / den**0.5 #* 5 ** 0.5
    # figure()
    # demo_distr(T5, theoretical = StudentTDistr(5), xmin = -1e2, xmax=1e2)
    # figure()
    # demo_distr(num, theoretical = NormalDistr(0, 5**0.5), xmin = -1e1, xmax=1e1)
    # figure()
    # demo_distr(den, theoretical = ChiSquareDistr(5), xmin = -1e1, xmax=1e1)

    #N1 = NormalDistr(0,1); demo_distr(N1, theoretical = N1, title = "Normal test")
    #N2 = N1 + 1; demo_distr(N2, theoretical = NormalDistr(1,1))
    #N2 = 1 + N2; demo_distr(N2, theoretical = NormalDistr(2,1))
    #N2 = N2 - 1; demo_distr(N2, theoretical = NormalDistr(1,1))
    #Nerr = N1 + "a"
    #N3 = N1 + N2; demo_distr(N3, theoretical = NormalDistr(1,sqrt(2)))
    #negN2 = -N2; negN2.plot()
    #N4 = N1 - N2; N4.plot()

    #N5 = N1 ** 2; demo_distr(N5)
    #N6 = N2 ** 2; demo_distr(N6)
    #N7 = DivDistr(N1, N1)
    #N8 = DivDistr(N2, N2)
    #N9 = MulDistr(N2, SumDistr(N2, N2))
    #N10 = SquareDistr(N1)
    #N11 = N5 + N5; demo_distr(N11)
    #N12 = N5 + N11; demo_distr(N12)
    #N13 = MulDistr(N1, N1)
    #N13prime = N1 * NormalDistr(0,1); demo_distr(N13prime)
    #N14 = 0.5 * N2
    #N15 = N2 * -1
    #N16 = NormalDistr(0,1) / NormalDistr(0,1); demo_distr(N16)
    #N17 = 2 - N1

    # figure()
    # N18 = 2 / N1; N18.plot()
    # N19 = (N18 + N18) / 2; N19.plot()
    # N20 = (N19 * 2 + N18) / 3; N20.plot()
    # N21 = (N20 * 3 + N18) / 4; N21.plot()

    # absolute value
    # figure()
    # N22 = abs(N1); N22.plot()
    # N23 = abs(N2); N23.plot()
    # N24 = N22 + N23; N24.plot()

    # figure()
    # #demo_distr(NormalDistr(0,1) / NormalDistr(0,1))
    # #figure()
    # N25 = atan(NormalDistr(0,1) / NormalDistr(0,1));
    # N26 = NormalDistr(0,1) / NormalDistr(0,1)
    # N27 = NormalDistr(0,1) * (1 / NormalDistr(0,1))
    # figure()
    # demo_distr(N25, theoretical = UniformDistr(-pi/2, pi/2), title = "atan(Cauchy) = atan(N(0,1)/N(0/1))", histogram = True)
    # figure()
    # demo_distr(N26, theoretical = lambda x: 1.0/pi/(1+x*x), title = "N(0,1) / N(0,1) == N(0,1) * (1 / N(0,1))", histogram = False)
    # demo_distr(N27, theoretical = lambda x: 1.0/pi/(1+x*x), title = "N(0,1) / N(0,1) == N(0,1) * (1 / N(0,1))", histogram = False)

    # # powers, exponents, logs
    # figure()
    #N26 = exp(N1); print N26; demo_distr(N26)
    #N27 = N26 + N26; demo_distr(N27)
    # #N28 = exp(NormalDistr(0,1) / NormalDistr(0,1)); demo_distr(N28)
    # #N29 = N28 + N28; demo_distr(N29)
    # N30 = 2 ** N1; print N30; demo_distr(N30)
    #N31 = log(abs(N1)); print N31; demo_distr(N31)
    # N32 = abs(N1) ** N1; demo_distr(N32, xmin=0, xmax = 3, ymax = 2)
    # figure()
    # u = UniformDistr(1,2)**UniformDistr(1,2)
    # u.plot()
    # u.hist()
    # figure()
    # u = UniformDistr(0,1)**UniformDistr(0,2)
    # demo_distr(u, xmin=0, xmax = 1, ymax = 3)

    # figure()
    # U1 = UniformDistr(1,3)
    # # UN1 = U1 / N2; demo_distr(UN1)
    # # UN2 = U1 + N2
    # # UN3 = U1 - UniformDistr(2,5)
    # # UN4 = FuncDistr(MulDistr(U1, U1), f = lambda x: x/3.0, f_inv = lambda x: 3*x, f_inv_deriv = lambda x: 3)
    # UN5 = UniformDistr(1,2) / UniformDistr(3,4); demo_distr(UN5)
    # UN7 = N2 * UniformDistr(9,11); demo_distr(UN7)
    # UN9 = UniformDistr(9,11) * N2; demo_distr(UN9)
    # UN11 = UniformDistr(-2,1) / UniformDistr(-2,1); demo_distr(UN11)
    # UN6 = atan(UN5); demo_distr(UN6)
    # UN10 = atan(UniformDistr(3,5)); demo_distr(UN10); print UN10

    # figure()
    # UN8 = UniformDistr(1,3) / UniformDistr(-2,1); demo_distr(UN8)
    # UN12 = (UN8 + UN8) / 2; demo_distr(UN12)
    # UN13 = (UN12 * 2 + UN8) / 3; demo_distr(UN13)
    # UN14 = (UN13 * 3 + UN8) / 4; demo_distr(UN14)
    # UN14_2 = (UN14 * 4 + UN8) / 5; demo_distr(UN14_2)
    # UN14_2 = (UN14 * 4 + UN8) / 5; demo_distr(UN14_2)
    # UN14_3 = (UN14_2 * 5 + UN8) / 6; demo_distr(UN14_3)
    # UN14_4 = (UN14_3 * 6 + UN8) / 7; demo_distr(UN14_4)
    # UN14_5 = (UN14_4 * 7 + UN8) / 8; demo_distr(UN14_5)

    # the slash distribution
    #figure()
    #UN15 = NormalDistr(0,1) / UniformDistr(0,1); demo_distr(UN15)
    #UN16 = (UN15 + UN15) / 2; demo_distr(UN16)
    #UN17 = (UN16 * 2 + UN15) / 3; demo_distr(UN17)
    #UN18 = (UN17 * 3 + UN15) / 4; demo_distr(UN18)
    #UN19 = (UN18 * 4 + UN15) / 5; demo_distr(UN19)
    #UN20 = (UN19 * 5 + UN15) / 6; demo_distr(UN20)

    # from Springer and others
    #figure()
    #CauchyDistr().plot()
    #demo_distr(CauchyDistr(), histogram = False)
    #UN21 = NormalDistr(0,1) / NormalDistr(0,1) #* UniformDistr(-1,1)
    #demo_distr(UN21, histogram = True)
    #UN21.summary()
    #UN22 = NormalDistr(-1,1) / NormalDistr(-1,1)# * UniformDistr(0,1)
    #demo_distr(UN22)
    #UN23 = NormalDistr(-1,1) / NormalDistr(-1,1) * UniformDistr(0,1)
    #figure()
    #UN23.plot()
    #UN23.hist(xmin=-2, xmax=2)
    #UN23_copy = NormalDistr(-1,1) / NormalDistr(-1,1) * UniformDistr(0,1)
    #UN24 = (UN23 + UN23_copy) / 2
    #figure()
    #UN24.plot()
    #demo_distr(UN24, xmin = -2, xmax = 2)
    # tails
    # !!!! does not work with /2
    #UN24.get_piecewise_pdf().plot_tails()

    #U = UniformDistr(0,1)
    #U_inv = InvDistr(U)
    #X = U * U_inv
    #Y = U / U
    #R = X.get_piecewise_pdf() - Y.get_piecewise_pdf()
    #figure()
    #R.plot()

    # Cauchy with params
    #demo_distr(CauchyDistr(gamma = 2, center = 1), histogram = True, xmin = -3, xmax = 5)

    # Gamma
    # figure()
    # demo_distr(GammaDistr(1,2), xmax = 20)
    # demo_distr(GammaDistr(2,2), xmax = 20)
    # demo_distr(GammaDistr(3,2), xmax = 20)
    # demo_distr(GammaDistr(5,1), xmax = 20)
    # demo_distr(GammaDistr(9,0.5), xmax = 20)
    # demo_distr(GammaDistr(0.5,2), xmax = 20, ymax = 1.2)
    # figure()
    # demo_distr(GammaDistr(1,1) + GammaDistr(1,1) + GammaDistr(1,1), theoretical = GammaDistr(3,1), xmax = 50)


    # Beta
    # figure()
    # print BetaDistr(alpha = 1, beta = 1)
    # demo_distr(BetaDistr(alpha = 1, beta = 1), theoretical = UniformDistr(0,1))
    # figure()
    # demo_distr(BetaDistr(alpha = 0.5, beta = 0.5), histogram = True, ymax = 3)
    # demo_distr(BetaDistr(alpha = 2, beta = 2), histogram = True, ymax = 3)
    # demo_distr(BetaDistr(alpha = 8, beta = 5), histogram = True, ymax = 3)

    # figure()
    # demo_distr(BetaDistr(alpha = 0.3, beta = 0.1), histogram = True, ymax = 3)
    # figure()
    # demo_distr(BetaDistr(alpha = 0.3, beta = 0.1) + BetaDistr(alpha = 0.3, beta = 0.1), ymax = 3)
    # figure()
    # demo_distr(BetaDistr(alpha = 0.3, beta = 0.1) + BetaDistr(alpha = 0.3, beta = 0.1) + BetaDistr(alpha = 0.3, beta = 0.1), ymax = 3)

    # figure()
    # bb = BetaDistr(alpha = 0.5, beta = 0.5) + BetaDistr(alpha = 0.5, beta = 0.5)
    # Xs, Ys =  bb.get_piecewise_pdf().segments[-1].f.getNodes()
    # #for x, y, xx in zip(Xs, Ys, bb.get_piecewise_pdf().segments[-1].f.Xs):
    # #    print repr(x), repr(xx), repr(y)
    # demo_distr(bb, histogram = True, ymax = 3)
    # figure()
    # bbb = bb + BetaDistr(alpha = 0.5, beta = 0.5)
    # demo_distr(bbb, histogram = True, ymax = 1)
    # figure()
    # bbbb = bbb + BetaDistr(alpha = 0.5, beta = 0.5)
    # demo_distr(bbbb, histogram = True, ymax = 1)
    # figure()
    # bbbbb = bbbb + BetaDistr(alpha = 0.5, beta = 0.5)
    # demo_distr(bbbbb, histogram = True, ymax = 1)

    # Pareto
    #figure()
    #p = ParetoDistr(1, 3)
    #demo_distr(p, xmin = 1, xmax = 1e20)

    #figure()
    #p2 = ParetoDistr(1.5)
    #demo_distr(p2, xmin = 1, xmax = 1e20)
    #figure()
    #p2.get_piecewise_cdf().plot(show_nodes = True, right=20)
    #figure()
    #demo_distr((p2+p2+p2+p2)/4.0, xmin = 1, xmax = 20)
    # demo_distr(p, xmin = 1, xmax = 20)
    # demo_distr(p+p, xmin = 1, xmax = 20)
    # figure()
    # demo_distr(log(p/3), xmin = 1, xmax = 20)
    # figure()
    # p2 = ParetoDistr(0.1)
    # demo_distr(p2, xmin = 1, xmax = 1e2, ymax = 0.1)
    # figure()
    # p2.get_piecewise_cdf().plot(show_nodes = True, right=5e1)
    # figure()
    # demo_distr((p2+p2)/2, xmin = 1, xmax = 1e2, ymax = 0.01)

    # c1 = ChiSquareDistr(1)
    # L = LevyDistr()
    # c = L**(-1)
    # figure()
    # demo_distr(L**(-1), theoretical = c1,xmin = 0, xmax = 1e1, ymax = 3)
    # figure()
    # demo_distr(1/c1, theoretical = L, xmin = 0, xmax = 10)
    # figure()
    # demo_distr(1/(L+L), theoretical = c1, xmin = 0, xmax = 1e1, ymax = 3)
    #
    # print c.get_piecewise_pdf()


    # ChiSquareDistr
    # c1 = ChiSquareDistr(1)
    # figure()
    # c2 = c1 + c1
    # demo_distr(c2, theoretical = ChiSquareDistr(2))
    # figure()
    # c3 = c2 + c1
    # demo_distr(c3, theoretical = ChiSquareDistr(3))
    # figure()
    # c4 = c3 + c1
    # demo_distr(c4, theoretical = ChiSquareDistr(4))
    # figure()
    # c5 = c4 + c1
    # demo_distr(c5, theoretical = ChiSquareDistr(5))

    # # Levy
    # l = LevyDistr()
    # figure()
    # demo_distr(l, xmin = 0, xmax = 10)
    # demo_distr(l+l, xmin = 0, xmax = 10)
    # demo_distr(l*l, xmin = 0, xmax = 10)
    # figure()
    # demo_distr(1 / ChiSquareDistr(1), xmin = 0, xmax = 10)
    # demo_distr(1 / l, xmin = 0, xmax = 10)

    # # Laplace
    # figure()
    # demo_distr(LaplaceDistr())
    # demo_distr(LaplaceDistr() + LaplaceDistr())
    # figure()
    # demo_distr(abs(LaplaceDistr()), theoretical = ExponentialDistr())
    # figure()
    # demo_distr(NormalDistr()*NormalDistr() + NormalDistr()*NormalDistr(), theoretical = LaplaceDistr())

    # Student t
    # figure()
    # demo_distr(StudentTDistr(3), xmin = -5, xmax = 5)
    # demo_distr(StudentTDistr(0.5), xmin = -5, xmax = 5)
    # demo_distr(StudentTDistr(100), xmin = -5, xmax = 5)
    # figure()
    # demo_distr(NormalDistr() / sqrt(ChiSquareDistr(3) / 3), theoretical = StudentTDistr(3), xmin = -5, xmax = 5)
    # figure()
    # demo_distr(NormalDistr() / (sqrt(ChiSquareDistr(3)) / sqrt(3.0)), theoretical = StudentTDistr(3), xmin = -5, xmax = 5)
    # figure()
    # demo_distr(NormalDistr() / (sqrt(ChiSquareDistr(3))) * sqrt(3.0), theoretical = StudentTDistr(3), xmin = -5, xmax = 5)

    # figure()
    # n = 4
    # T10 = (NormalDistr() / sqrt(ChiSquareDistr(n))) * n ** 0.5
    # demo_distr(T10, theoretical = StudentTDistr(n), xmin = -1e10, xmax=1e10)
    # figure()
    # def test_student(n):
    #     N1 = NormalDistr()
    #     print "================================================num===="
    #     num = N1
    #     for i in range(n-1):
    #         num  += N1
    #     num.get_piecewise_pdf()
    #     C1 = ChiSquareDistr(1)
    #     print "================================================den===="
    #     den = C1
    #     for i in range(n-1):
    #         den  += C1
    #     Tn = num / den**0.5 #* 5 ** 0.5
    #
    #     print Tn.get_piecewise_pdf().segments[0].f.vl.getNodes()
    #     figure()
    #     demo_distr(num, theoretical = NormalDistr(0, n**0.5), xmin = -1e2, xmax=1e2)
    #     figure()
    #     demo_distr(den, theoretical = ChiSquareDistr(n), xmin = 0, xmax=1e1)
    #     figure()
    #     demo_distr(Tn, theoretical = StudentTDistr(n), xmin = -1e1, xmax=1e1)
    #      return num, den, Tn
    # test_student(5)
    # test_student(4)

    # demo_distr(NormalDistr(1,1) * NormalDistr(1,1))
    # figure()
    # demo_distr(NormalDistr(10,1) * NormalDistr(10,1))
    # figure()
    # demo_distr(NormalDistr(100,1) * NormalDistr(100,1))

    # figure()
    # demo_distr(FDistr(2, 2), xmax = 10)
    # figure()
    # demo_distr(FDistr(4, 5), xmax = 10)
    # figure()
    # demo_distr(FDistr(2, 2) + FDistr(4, 5) + FDistr(1, 1), xmax = 10)

    # figure()
    # df1, df2, df3 = 2, 101,40
    # c1 =ChiSquareDistr(df1)
    # c2 =ChiSquareDistr(df2)
    # c3 =ChiSquareDistr(df3)
    # c4 =ChiSquareDistr(df3)
    # d = c1 + c2
    # c1.summary()
    # c2.summary()
    # c3.summary()
    # d.summary()
    # d.plot(right=1e4)
    # demo_distr(d, theoretical = ChiSquareDistr(df1  + df2 ), xmax = 1e4)
    #
    # show()
    # 0/0
    #
    # # figure()
    # # demo_distr(SemicircleDistr())
    # figure()
    # demo_distr(SemicircleDistr() + SemicircleDistr())
    # # figure()
    # # demo_distr(SemicircleDistr() + SemicircleDistr() + SemicircleDistr())
    # figure()
    # demo_distr(SemicircleDistr() + BetaDistr(0.2, 0.9))
    #
    # figure()
    # demo_distr(SemicircleDistr())
    # figure()
    # demo_distr(BetaDistr(0.2, 0.9))
    #
    # figure()
    # F = SemicircleDistr() + BetaDistr(0.2, 0.9)
    # c = F.get_piecewise_cdf()
    # F.plot()
    # c.plot()
    # figure()
    # #hist(F.rand_invcdf(10000))
    # F.plot()

    # # demo_distr(SemicircleDistr())
    # figure()
    # demo_distr(SemicircleDistr() + SemicircleDistr())
    # figure()
    # demo_distr(SemicircleDistr() + SemicircleDistr() + SemicircleDistr())
    # # this does not work (at least not yet ;)
    # figure()
    # dd = SemicircleDistr() + BetaDistr(0.3, 0.1)
    # demo_distr(dd)
    # figure()
    # dd = SemicircleDistr() + BetaDistr(0.5, 0.5)
    # demo_distr(dd)


    #demo_distr(N1)
    #demo_distr(N2)
    #demo_distr(UN2)
    #demo_distr(N4)
    #demo_distr(N5)
    #demo_distr(N6)
    #demo_distr(N7)
    #demo_distr(N8)
    #demo_distr(N9)
    #demo_distr(N10);demo_distr(N13) # Example: X**2 the same as X*X with our semantics
    #demo_distr(N10);demo_distr(N13prime) # Example: X**2 not the same as X*X.copy()
    #demo_distr(N11)
    #demo_distr(N14);demo_distr(N15) # multiply by a constant
    #demo_distr(U1)
    #demo_distr(UN1)
    #demo_distr(UN2)
    #demo_distr(UN3);print UN3.breaks
    #demo_distr(UN4)
    #demo_distr(UN6)
    #demo_distr(UN7)
    #demo_distr(UN8, xmin=-6, xmax=6) # my favorite distr
    #demo_distr(UN9)
    #demo_distr(UN10);print UN10.breaks
    #demo_distr(UN11, xmax = 4)
    #demo_distr(UN1, xmin = -10, xmax = 10);print UN1.breaks
    #demo_distr(UI1); print "U1.err =",UI1.err
    #demo_distr(UI2)




    ## another example: http://www.physicsforums.com/showthread.php?t=75889
    ## probability density of two resistors in parallel XY/(X+Y)
    ##
    ## XY an X+Y are not independent: we are not yet ready to handle this
    #L = 40; U = 70
    #X = uniformDistr(100,120)
    #Y = uniformDistr(100,120)
    #N = mulDistr(X,Y)
    #D = sumDistr(X,Y)
    #Ni = interpolatedDistr(N)
    #Di = interpolatedDistr(D)
    #R = divDistr(Ni, Di)
    #Ri = interpolatedDistr(R)
    #print integrate(lambda x: Ri.PDF(x), limit=100)
    #demo_distr(Ri, L, U)
    #xlim(L, U)


    # Distrete

    #
    # figure()
    # d = DiscreteDistr(xi =[0, 1], pi = [0.2, 0.8])
    #
    # b5 = d + d + d + d + d
    # b5.plot()
    # b5.get_piecewise_cdf().plot()
    # b5.hist()
    #
    # d = DiscreteDistr(xi =[1, 2], pi = [0.2, 0.8])
    # U = UniformDistr(0,2)
    # A1 = d + U
    # A2 = d * U
    # A3 = d / U
    # A4 = U / d
    # figure()
    #
    # A1.plot()
    # A2.plot()
    # A3.plot()
    # A4.plot()
    # A1.hist(A1)
    # A2.hist(A2)
    # A3.hist(A3, xmin=0.5, xmax=5)
    # A4.hist(A4)
    #
    # figure()
    # A1.get_piecewise_cdf().plot()
    # A2.get_piecewise_cdf().plot()
    # A3.get_piecewise_cdf().plot()
    # A4.get_piecewise_cdf().plot()

    # figure()
    # g = GumbelDistr()
    # demo_distr(g)
    # figure()
    # g2 = GumbelDistr(3, 5)
    # demo_distr(g2)
    # figure()
    # gg = g + g2
    # demo_distr(gg)

    # figure()
    # f = FrechetDistr(s=3)
    # demo_distr(f, xmax=10)
    show()
