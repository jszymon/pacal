"""Base distribution classes.  Operations on distributions."""

from __future__ import print_function

import numbers
from functools import partial

import numpy
from numpy import array, zeros_like, ones_like, unique, concatenate, isscalar, isfinite
from numpy import sqrt, pi, arctan, tan, asfarray, asarray, zeros, Inf, NaN
#from numpy import sin, cos, tan,
from numpy import arcsin, arccos
from numpy.random import uniform
from numpy import minimum, maximum
from numpy import hstack, cumsum, searchsorted
from numpy import histogram

from pacal.utils import Inf

from pylab import bar, plot

import time
import traceback

from . import params
from pacal.indeparith import conv, convprod, convdiv, convmin, convmax
from pacal.segments import PiecewiseFunction, PiecewiseDistribution, DiracSegment, ConstSegment
from pacal.rv import *


def _mgf_fun(distr, t):
    if isscalar(t):
        return distr.meanf(f=lambda x: exp(t*x))
    else:
        y = zeros_like(t)
        for i in range(len(t)):
            y[i] = _mgf_fun(distr, t[i])
        return y

#class Distr(object):
#    def __init__(self, parents = [], indep = None):
class Distr(RV):
    def __init__(self, parents = [], indep = True, sym = None):
        super(Distr, self).__init__(parents, sym)
        # indep = True means the distribution is treated as
        # independent from all others.  For examples this results in
        # X+X != 2X.  This currently only affects random number
        # generation and histograms.  This default will likely change
        # in the future.
        if indep is not None:
            self.indep = indep
        else:
            self.indep = params.general.distr.independent
        self.piecewise_pdf = None # PDF represented as piecewise
                                  # function, usually interpolated.
        self.piecewise_cdf = None # CDF represented as piecewise
                                  # function
        self.piecewise_cdf_interp = None    # CDF represented as  interpolated.
                                            # piecewise function
        self.piecewise_ccdf = None # complementary CDF represented as piecewise
                                   # function
        self.piecewise_ccdf_interp = None   # complementary CDF represented
                                            # as interpolated piecewise function
        # check dependencies in parents
        if params.general.warn_on_dependent:
            anc = set()
            for p in self.parents:

                panc = p.getAncestorIDs()
                if panc & anc:
                    print("Warning: arguments treated as independent")
                    break
                anc.update(panc)
    def getName(self):
        """return a string representation of PDF."""
        return "D"
    def get_piecewise_pdf(self):
        """return PDF function as a PiecewiseDistribution object"""
        if self.piecewise_pdf is None:
            self.init_piecewise_pdf()
        return self.piecewise_pdf
    def get_piecewise_cdf(self):
        """return CDF function as a CumulativePiecewiseFunction object"""
        if self.piecewise_cdf is None:
            self.piecewise_cdf = self.get_piecewise_pdf().cumint()      # integrals are computed directly - much slower
            #self.piecewise_cdf_interp = self.get_piecewise_cdf().toInterpolated()   # interpolated version - much faster
        return self.piecewise_cdf
    def get_piecewise_cdf_interp(self):
        """return, CDF function as CumulativePiecewiseFunction object

        This is interpolated version of piecewise_cdf, much faster
        as specially for random number greneration"""
        if self.piecewise_cdf_interp is None:
            self.piecewise_cdf_interp = self.get_piecewise_cdf().toInterpolated()   # interpolated version - much faster
        return self.piecewise_cdf_interp
    def get_piecewise_ccdf(self):
        """return, CDF function, as CumulativePiecewiseFunction object"""
        if self.piecewise_ccdf is None:
            self.piecewise_ccdf = 1 - self.get_piecewise_cdf()
            # integrals are computed directly - much slower
            #self.piecewise_cdf_interp = self.get_piecewise_cdf().toInterpolated()   # interpolated version - much faster
        return self.piecewise_ccdf
    def get_piecewise_ccdf_interp(self):
        """return, CDF function as CumulativePiecewiseFunction object

        This is interpolated version of piecewise_cdf, much faster
        as specially for random number greneration"""
        if self.piecewise_ccdf_interp is None:
            self.piecewise_ccdf_interp = self.get_piecewise_ccdf().toInterpolated()   # interpolated version - much faster
        return self.piecewise_ccdf_interp

    def init_piecewise_pdf(self):
        """Initialize the pdf represented as a piecewise function.

        This method should be overridden by subclasses."""
        raise NotImplementedError()
    def get_piecewise_invcdf(self, use_interpolated=True):
        """return, inverse CDF function, as PiecewiseFunction object"""
        if use_interpolated:
            invcdf  = self.get_piecewise_cdf_interp().invfun(use_interpolated=use_interpolated, rangeY=None)
        else:
            invcdf  = self.get_piecewise_cdf().invfun(use_interpolated=use_interpolated, rangeY=None)
        return invcdf

    def pdf(self,x):
        return self.get_piecewise_pdf()(x)
    def cdf(self,x):
        """Cumulative piecewise function."""
        return self.get_piecewise_cdf()(x)
    def ccdf(self,x):
        """Complementary cumulative piecewise function.
        Not implemented yet. """
        #pass
        # TODO temporary solution, to remove
        return self.get_piecewise_ccdf()(x) #TODO implement it

    def ccdf_value(self,x):
        """Complementary cumulative distribution function.

        This methods  gives better accuracy than 1-cdf(x) in neighborhood of
        right infinity. It works properly only with scalars."""
        segments = self.get_piecewise_pdf().segments
        seg = segments[-1]
        if x<=seg.a or not seg.isPInf():
            return 1-self.get_piecewise_cdf()(x)
        else:
            return seg.integrate(x)
    def log_pdf(self,x):
        return log(self.pdf())
    def mean(self):
        """Mean of the distribution."""
        return self.get_piecewise_pdf().mean()
    def meanf(self, f):
        """Mean value of f(X) w.r.t. the distribution."""
        return self.get_piecewise_pdf().meanf(f=f)
    def moment(self, k, c=None):
        """Moment about c of k-th order of distribution. """
        if c is None:
            c = self.mean()
        def f(x, c=c, k=k):
            return (x-c)**k
        return self.get_piecewise_pdf().meanf(f=f)
    def skewness(self):
        if not self.var() == 0:
            return self.moment(3, self.mean()) / self.var()**1.5
        else:
            return NaN
    def kurtosis(self):
        if not self.var() == 0:
            return self.moment(4, self.mean()) / self.var()**2
        else:
            return NaN
    def mgf(self):
        return PiecewiseFunction(fun=partial(_mgf_fun, self), breakPoints=self.get_piecewise_pdf().getBreaks())
    def cf(self):
        # TODO
        pass

    def std(self):
        """Mean of the distribution."""
        return self.get_piecewise_pdf().std()
    def var(self):
        """Variance of the distribution."""
        return self.get_piecewise_pdf().var()
    def medianad(self):
        """Median absolute dispersion of the distribution."""
        return self.get_piecewise_pdf().medianad()
    def median(self):
        """Median of the distribution."""
        return self.get_piecewise_pdf().median()
    def entropy(self):
        """Median of the distribution."""
        return self.get_piecewise_pdf().entropy()
    def KL_dist(self,other):
        """Median of the distribution."""
        return self.get_piecewise_pdf().KL_distance(other.get_piecewise_pdf())
    def L2_dist(self,other):
        """Median of the distribution."""
        return self.get_piecewise_pdf().L2_distance(other.get_piecewise_pdf())
#    #TODO fix this to use only rv.range() in all cases
    def range_(self, lazy=True):
        """Range of the distribution."""
        return self.get_piecewise_pdf().range()

    def iqrange(self, level=0.025):
        """Inter-quantile range of the distribution."""
        return self.quantile(1-level) - self.quantile(level)
        #clevel = 1 - level
        #return self.quantile(1-clevel/2.0) - self.quantile(clevel/2.0)

    def is_nonneg(self):
        """Check whether distribution is positive definite."""
        return self.get_piecewise_pdf().isNonneg()
    def quantile(self, y):
        """The quantile function - inverse cumulative distribution
        function."""
        return self.get_piecewise_cdf().inverse(y)
    def ci(self, p = 0.05):
        """Confidence interval.

        Keyword arguments:
        p : significance level"""
        return (self.quantile(p/2), self.quantile(1-p/2.0))
    def interval(self, p = 0.95):
        """Returns symmetrical interval that supports
        p-percent of probability mass.

        Keyword arguments:
        p : significance level"""
        p_lim = (1.0 - p) / 2.0
        return self.quantile(p_lim), self.quantile(1.0 - p_lim)
    def tailexp(self):
        """Left and right tail exponent estimates"""
        return self.get_piecewise_pdf().tailexp()

    def mode(self):
        """Mode of distribution."""
        return self.get_piecewise_pdf().maximum()[0]

    def int_error(self):
        """L_1 error for testing of accuracy"""
        return 1-self.get_piecewise_pdf().integrate()
    def interp_error_by_segment(self):
        """Estimated error of interpolation for each segment."""
        return self.get_piecewise_pdf().getInterpErrors()
    def interp_error(self):
        """Estimated maximum error of interpolation."""
        errs = [e[2] for e in self.get_piecewise_pdf().getInterpErrors()]
        return max(errs)
    def summary_map(self):
        r = {}
        r['mean'] = self.mean()
        r['std'] = self.std()
        r['var'] = self.var()
        r['skewness'] = self.skewness()
        r['kurtosis'] = self.kurtosis()
        r['entropy'] = self.entropy()
        r['range'] = self.get_piecewise_pdf().range()
        r['int_err'] = self.int_error()
        r['tailexp'] = self.tailexp()
        r['mode'] = self.mode()
        #r['interp_errs'] = self.getInterpErrors()
        try:
            r['mode'] = self.mode()
            r['median'] = self.median()
            r['iqrange(0.025)'] = self.iqrange()
            r['medianad'] = self.medianad()
            r['interval(0.95)'] = self.interval()
        except Exception as e:
            traceback.print_exc()
        return r
    def summary(self, show_moments=False):
        """Summary statistics for a given distribution."""
        print("============= summary =============")
        #print self.get_piecewise_pdf()
        #t0  = time.time()
        summ = self.summary_map()
        print(" ", self.getName())
        for i in ['mean', 'var', 'skewness', 'kurtosis', 'entropy', 'median', 'mode', 'medianad', 'iqrange(0.025)', 'interval(0.95)',  'range',  'tailexp', 'int_err']:
            if i in summ:
                print('{0:{align}20}'.format(i, align = '>'), " = ", repr(summ[i]))
            else:
                print("---", i)
        if show_moments:
            print("      moments:")
            for i in range(11):
                mi = self.moment(i,0)
                print('{0:{align}20}'.format(i, align = '>'), " = ", repr(mi))
        #print "=====", time.time() - t0, "sec."
    def rand_raw(self, n = None):
        """Generates random numbers without tracking dependencies.

        This method will be implemented in subclasses implementing
        specific distributions.  Not intended to be used directly."""
        return None
    def rand_invcdf(self, n = None, use_interpolated=True):
        """Generates random numbers through the inverse cumulative
        distribution function.

        May use interpolated inverse of cdf for speed."""
        y = uniform(0, 1, n)
        return self.get_piecewise_invcdf(use_interpolated=use_interpolated)(y)
    def rand(self, n = None, cache = None):
        """Generates random numbers while tracking dependencies.

        if n is None, return a scalar, otherwise, an array of given
        size."""
        if self.indep:
            return self.rand_raw(n)
        if cache is None:
            cache = {}
        if self.id() not in cache:
            cache[self.id()] = self.rand_raw(n)
        return cache[self.id()]
    def plot(self, *args, **kwargs):
        """Plot of PDF.

        Keyword arguments:
        xmin -- minimum x range
        xmax -- maximum x range
        other of pylab/plot **kvargs
        """
        self.get_piecewise_pdf().plot(*args, **kwargs)
    def hist(self, n = 1000000, xmin = None, xmax = None, bins = 50, max_samp = None,
             alpha=0.25, color="lightgray", edgecolor="darkgray", **kwargs):
        """Histogram of PDF.

        Keyword arguments:
        n -- number of points
        bins -- number of bins
        xmin -- minimum x range
        xmax -- maximum x range

        Histogram show frequencies rather then cardinalities thus it can be
        compared with PDF function in continuous case. When xmin, xmax
        are defined then conditional histogram is presented."""
        if max_samp is None:
            max_samp = 100 * n
        if xmin is None and xmax is None:
            X = self.rand(n, None)
            allDrawn = len(X)
        else:
            X = []
            allDrawn = 0
            while len(X) < n and allDrawn < max_samp:
                x = self.rand(n - len(X))
                allDrawn = allDrawn + len(x)
                if xmin is not None:
                    x = x[(xmin <= x)]
                if xmax is not None:
                    x = x[(x <= xmax)]
                X = hstack([X, x])
        #hist(X, bins, normed = True, alpha = 0.25, **kwargs)
        dw = (X.max() - X.min()) / bins
        if dw == 0:
            dw = 1
        w = (float(n)/float(allDrawn)) / n / dw
        counts, binx = histogram(X, bins)
        width = binx[1] - binx[0]
        if width == 0:
            width = X.max() * 0.01
        for c, b in zip(counts, binx):
            bar(b, float(c) * w, width = width, alpha=alpha, align="edge", color=color,
                edgecolor=edgecolor, **kwargs)
    def five_number_summary(self):
        m = self.median()
        iqr = self.iqrange(0.25)
        c = self.interval(0.95)
        r = self.range()
        if isfinite(r[0]) and isfinite(r[1]):
            return [r[0], c[0], m, c[1], r[1]]
        elif isfinite(r[0]):
            return [r[0], c[0], m, c[1], c[1] + 1.5 * iqr]
        elif isfinite(r[1]):
            return [c[0] - 1.5 * iqr, c[0], m, c[1], r[1]]
        else:
            return [c[0] - 1.5 * iqr, c[0], m, c[1], c[1] + 1.5 * iqr]

    def boxplot(self, pos=1, width=0.3, useci=None, showMean=True, vertical=True, color="k", label=None, **kwargs):
        five = self.five_number_summary()
        if useci is not None:
            c = self.interval(useci)
            five[0] = c[0]
            five[-1] = c[1]
        #if label is None:
        #    label = "" + str(self.__class__.__name__)
        if vertical:
            plot([pos-width, pos+width], [five[1], five[1]], color=color, label=label, **kwargs)
            plot([pos-width, pos+width], [five[2], five[2]], color=color, **kwargs)
            plot([pos-width, pos+width], [five[3], five[3]], color=color, **kwargs)
            plot([pos-width/2, pos+width/2], [five[0], five[0]], color=color, **kwargs)
            plot([pos-width/2, pos+width/2], [five[4], five[4]], color=color, **kwargs)
            plot([pos-width, pos-width], [five[1], five[3]], color=color, **kwargs)
            plot([pos+width, pos+width], [five[1], five[3]], color=color, **kwargs)
            plot([pos, pos], [five[0], five[1]], '--', color=color, **kwargs)
            plot([pos, pos], [five[3], five[4]], '--', color=color, **kwargs)
            if showMean:
                m = self.mean()
                plot([pos-width, pos+width], [m, m], '--', color=color, **kwargs)

    def __call__(self, x):
        """Overload function calls."""
        return self.pdf(x)
    # overload arithmetic operators
    def __neg__(self):
        """Overload negation distribution of -X."""
        return ShiftedScaledDistr(self, scale = -1)
    def __abs__(self):
        """Overload abs: distribution of abs(X)."""
        return AbsDistr(self)
    def __add__(self, d):
        """Overload sum: distribution of X+Y."""
        if isinstance(d, Distr):
            return SumDistr(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, shift = d)
        return NotImplemented
    def __radd__(self, d):
        """Overload sum with real number: distribution of X+r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, shift = d)
        return NotImplemented
    def __sub__(self, d):
        """Overload subtraction: distribution of X-Y."""
        if isinstance(d, Distr):
            return SubDistr(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, shift = -d)
        return NotImplemented
    def __rsub__(self, d):
        """Overload subtraction with real number: distribution of X-r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, scale = -1, shift = d)
        return NotImplemented
    def __mul__(self, d):
        """Overload multiplication: distribution of X*Y."""
        if isinstance(d, Distr):
            return MulDistr(self, d)
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledDistr(self, scale = d)
        return NotImplemented
    def __rmul__(self, d):
        """Overload multiplication by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                if d == 1:
                    return self
                return ShiftedScaledDistr(self, scale = d)
        return NotImplemented
    def __truediv__(self, d):
        """Overload division: distribution of X/r."""
        if isinstance(d, Distr):
            return DivDistr(self, d)
        if isinstance(d, numbers.Real):
            if d == 1:
                return self
            return ShiftedScaledDistr(self, scale = 1.0 / d)
        return NotImplemented
    def __div__(self, d):
        """Overload python2 division: distribution of X/r."""
        if isinstance(d, Distr):
            return DivDistr(self, d)
        if isinstance(d, numbers.Real):
            if d == 1:
                return self
            return ShiftedScaledDistr(self, scale = 1.0 / d)
        return NotImplemented
    def __rtruediv__(self, d):
        """Overload division or real by distribution number: distribution of r / X."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            return d * InvDistr(self)
        return NotImplemented
    def __rdiv__(self, d):
        """Python2 version."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            return d * InvDistr(self)
        return NotImplemented
    def __pow__(self, d):
        """Overload power: distribution of X**Y,
        and special cases: X**(-1), X**2, X**0. X must be positive definite."""
        if isinstance(d, Distr):
            return ExpDistr(MulDistr(LogDistr(self), d))
        if isinstance(d, numbers.Real):
            if d == 0:
                return 1
            elif d == 1:
                return self
            elif d == -1:
                return InvDistr(self)
            elif d == 2:
                return SquareDistr(self)
            else:
                return ExpDistr(ShiftedScaledDistr(LogDistr(self), scale = d))
                #return PowDistr(self, alpha = d)
        return NotImplemented
    def __rpow__(self, x):
        """Overload power: distribution of X**r"""
        if isinstance(x, numbers.Real):
            if x == 0:
                return 0
            if x == 1:
                return 1
            if x < 0:
                raise ValueError()
            return ExpDistr(ShiftedScaledDistr(self, scale = numpy.log(x)))
        return NotImplemented
    def __or__(self, restriction):
        """Overload or: Conditional distribution """
        if isinstance(restriction, Condition):
            if isinstance(restriction, Lt):
                assert self.range_()[0]<restriction.U, "Upper value out of range"
                return CondLtDistr(self, restriction.U)
            if isinstance(restriction, Gt):
                assert self.range_()[1]>restriction.L, "Lower value out of range"+str(self.range_())+str(self)
                return CondGtDistr(self, restriction.L)
            if isinstance(restriction, Between):
                return CondBetweenDistr(self, restriction.L, restriction.U)
                #return CondLtDistr(CondGtDistr(self, restriction.L), restriction.U)
        return NotImplemented

def _wrapped_name(d, incl_classes = None):
    """Return name of d wrapped in parentheses if necessary"""
    d_name = d.getName()
    if incl_classes is not None:
        if isinstance(d, tuple(incl_classes)):
            d_name = "(" + d_name + ")"
    elif isinstance(d, OpDistr) and not isinstance(d, (FuncDistr, SquareDistr)):
        d_name = "(" + d_name + ")"
    return d_name

class OpDistr(Distr):
    """Base class for operations on distributions.

    Currently only does caching for random number generation."""
    def rand(self, n = 1, cache = None):
        if self.indep:
            return self.rand_op(n, None)
        if cache is None:
            cache = {}
        if self.id() not in cache:
            cache[self.id()] = self.rand_op(n, cache)
        return cache[self.id()]

class FuncDistr(FuncRV, OpDistr):
    """Injective function of random variable"""
    def __init__(self, d, pole_at_zero = False, fname = "f", sym = None):
        super(FuncDistr, self).__init__(d, sym = sym, fname = fname)
        self.pole_at_zero = pole_at_zero
# functions need to be implemented directly (not passed to constructor)
#        self.f = f
#        self.f_inv = f_inv
#        self.f_inv_deriv = f_inv_deriv

    # functions need to be implemented directly (not passed to constructor)
    def f(self, x):
        pass

    def f_inv(self, x):
        pass

    def f_inv_deriv(self, x):
        pass
    def pdf(self, x):
        y = self.d.pdf(self.f_inv(x)) * abs(self.f_inv_deriv(x))
        if isscalar(x):
            if not isfinite(self.f):
                y = 0
        else:
            mask = isfinite(y)
            y[~mask] = 0
        return y
    def rand_op(self, n, cache):
        return self.f(self.d.rand(n, cache))
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyComposition(self.f, self.f_inv, self.f_inv_deriv, pole_at_zero = self.pole_at_zero)

class ShiftedScaledDistr(ShiftedScaledRV, OpDistr):
    def __init__(self, d, shift = 0, scale = 1):
        assert(scale != 0)
        super(ShiftedScaledDistr, self).__init__(d, shift=shift, scale=scale)
        if self.scale is 1:
            self._1_scale = 1
        else:
            self._1_scale = 1.0 / scale
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyShiftedAndScaled(self.shift, self.scale)
    def pdf(self, x):
        return abs(self._1_scale) * self.d.pdf((x - self.shift) * self._1_scale)
    def rand_op(self, n, cache):
        return self.scale * self.d.rand(n, cache) + self.shift

def _one_over_abs(x):
    return 1.0/abs(x)
class ExpDistr(FuncDistr):
    """Exponent of a random variable"""
    def __init__(self, d):
        super(ExpDistr, self).__init__(d, pole_at_zero = True,
                                       fname = "exp")
    def f(self, x):
        return numpy.exp(x)
    def f_inv(self, x):
        return numpy.log(x)
    def f_inv_deriv(self, x):
        return 1.0/abs(x)
    def is_nonneg(self):
        return True
def exp(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return ExpDistr(d)
    if isinstance(d, RV):
        return ExpRV(d)
    return numpy.exp(d)

class LogDistr(FuncDistr):
    """Natural logarithm of a random variable"""
    def __init__(self, d):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        super(LogDistr, self).__init__(d, pole_at_zero= True, fname = "log")
    def f(self, x):
        return numpy.log(x)
    def f_inv(self, x):
        return numpy.exp(x)
    def f_inv_deriv(self, x):
        return numpy.exp(x)
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyLogComposition(self.f, self.f_inv, self.f_inv_deriv, pole_at_zero = self.pole_at_zero)

def log(d):
    """Overload the log function."""
    if isinstance(d, Distr):
        return LogDistr(d)
    if isinstance(d, RV):
        return LogRV(d)
    return numpy.log(d)

def sign(d):
    """Overload sign: distribution of sign(X)."""
    if isinstance(d, Distr):
        return SignDistr(d)
    if isinstance(d, RV):
        return SignRV(d)
    return numpy.sign(d)

class AtanDistr(FuncDistr):
    """Arcus tangent of a random variable"""
    def __init__(self, d):
        super(AtanDistr, self).__init__(d, pole_at_zero= False, fname ="atan")

    def f(self, x):
        return numpy.arctan(x)
    def f_inv(self, x):
        if isscalar(x):
            if x <= -pi/2 or x >= pi/2:
                y = 0
            else:
                y = numpy.tan(x)
        else:
            mask = (x > -pi/2) & (x < pi/2)
            y = zeros_like(asfarray(x))
            y[mask] = numpy.tan(x[mask])
        return y
    def f_inv_deriv(self, x):
        if isscalar(x):
            if x <= -pi/2 or x >= pi/2:
                y = 0
            else:
                y = 1 + numpy.tan(x)**2
        else:
            mask = (x > -pi/2) & (x < pi/2)
            y = zeros_like(asfarray(x))
            y[mask] = 1 + numpy.tan(x[mask])**2
        return y

def atan(d):
    """Overload the atan function."""
    if isinstance(d, Distr):
        return AtanDistr(d)
    if isinstance(d, RV):
        return AtanRV(d)
    return numpy.arctan(d)

class TanhDistr(FuncDistr):
    """Hyperbolic tangent of a random variable"""
    def __init__(self, d):
        super(TanhDistr, self).__init__(d, pole_at_zero= False, fname ="tanh")

    def f(self, x):
        return numpy.tanh(x)
    def f_inv(self, x):
        if isscalar(x):
            if x <= -1 or x >= 1:
                y = 0
            else:
                y = 0.5 * numpy.log((1.0+x) / (1.0-x))
        else:
            mask = (x > -1) & (x < 1)
            y = zeros_like(asfarray(x))
            y[mask] = 0.5 * numpy.log((1.0+x[mask]) / (1.0-x[mask]))
        return y
    def f_inv_deriv(self, x):
        if isscalar(x):
            if x <= -1 or x >= 1:
                y = 0
            else:
                y = 1.0 / (1 - x**2)
        else:
            mask = (x > -1) & (x < 1)
            y = zeros_like(asfarray(x))
            y[mask] = 1.0 / (1 - x[mask]**2)
        return y

def tanh(d):
    """Overload the tanh function."""
    if isinstance(d, Distr):
        return TanhDistr(d)
    if isinstance(d, RV):
        return TanhRV(d)
    return numpy.tanh(d)


class SigmoidDistr(FuncDistr):
    """Sigmoid function of a random variable"""
    def __init__(self, d):
        super(SigmoidDistr, self).__init__(d, pole_at_zero= False, fname ="tanh")

    def f(self, x):
        if isscalar(x):
            if x > 0:
                y = 1.0 / (1.0 + numpy.exp(-x))
            else:
                ex = numpy.exp(x)
                y =  ex / (ex + 1.0)
        else:
            y = zeros_like(asfarray(x))
            mask = (x > 0)
            y[mask] = 1.0 / (1.0 + numpy.exp(-x[mask]))
            ex = numpy.exp(x[~mask])
            y[~mask] = ex / (ex + 1.0)
        return y
    def f_inv(self, x):
        if isscalar(x):
            if x <= -1 or x >= 1:
                y = 0
            else:
                y = numpy.log(x / (1.0-x))
        else:
            mask = (x > -1) & (x < 1)
            y = zeros_like(asfarray(x))
            y[mask] = numpy.log(x[mask] / (1.0-x[mask]))
        return y
    def f_inv_deriv(self, x):
        if isscalar(x):
            if x <= -1 or x >= 1:
                y = 0
            else:
                y = 1.0 / (x*(1 - x))
        else:
            mask = (x > -1) & (x < 1)
            y = zeros_like(asfarray(x))
            y[mask] = 1.0 / (x[mask] * (1 - x[mask]))
        return y

def sigmoid(d):
    """Overload the tanh function."""
    if isinstance(d, Distr):
        return SigmoidDistr(d)
    if isscalar(x):
        if x > 0:
            y = 1.0 / (1.0 + numpy.exp(-x))
        else:
            ex = numpy.exp(x)
            y =  ex / (ex + 1.0)
    else:
        y = zeros_like(asfarray(x))
        mask = (x > 0)
        y[mask] = 1.0 / (1.0 + numpy.exp(-x[mask]))
        ex = numpy.exp(x[~mask])
        y[~mask] = ex / (ex + 1.0)
    return y


class InvDistr(InvRV, OpDistr):
    """Inverse of random variable."""
    def __init__(self, d):
        super(InvDistr, self).__init__(d)
        self.pole_at_zero = False
    def pdf(self, x):
        if isscalar(x):
            y = self.d.pdf(1.0/x)/x**2
        else:
            y = zeros_like(asfarray(x))
            mask = x != 0
            y[mask] = y = self.d.pdf(1.0/x[mask])/x[mask]**2
        return y
    def rand_op(self, n, cache):
        return 1.0/self.d.rand(n, cache)
    @staticmethod
    def f_(x):
        if isscalar(x):
            if x != 0:
                y = 1.0 / x
            else:
                y = Inf # TODO: put nan here
        else:
            mask = (x != 0.0)
            y = zeros_like(asfarray(x))
            y[mask] = 1.0 / x[mask]  # to powoduje bledy w odwrotnosci
            #y = 1.0 / x
        return y
    @staticmethod
    def f_inv_deriv(x):
        if isscalar(x):
            y = 1/x**2
        else:
            mask = (x != 0.0)
            y = zeros_like(asfarray(x))
            y[mask] = 1/(x[mask])**2
        return y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyProbInverse(pole_at_zero = self.pole_at_zero)

# unused !!!
class PowDistr(PowRV, OpDistr):
    """Inverse of random variable."""
    def __init__(self, d, alpha = 1):
        #super(PowDistr, self).__init__([d], self.f_, self.f_inv, self.f_inv_deriv, pole_at_zero = alpha > 1, fname="pow")
        super(PowDistr, self).__init__([d], pole_at_zero = alpha > 1, fname="pow")
        self.d = d
        self.alpha = alpha
        self.alpha_inv = 1.0 / alpha
        self.exp_deriv = self.alpha_inv - 1.0
    def pdf(self, x):
        if isscalar(x):
            y = self.d.pdf(1.0/x)/x**2
        else:
            y = zeros_like(asfarray(x))
            mask = x != 0
            y[mask] = y = self.d.pdf(1.0/x[mask])/x[mask]**2
        return y
    def __str__(self):
        return "#{0}**{1}".format(id(self.d1), self.alpha)
    def getName(self):
        return "{0}**{1}".format(_wrapped_name(self.d), self.alpha)
    def rand_op(self, n, cache):
        return self.d.rand(n, cache) ** self.alpha
    def f_(self, x):
        if isscalar(x):
            if x != 0:
                y = x ** (self.alpha)
            else:
                y = 0 # TODO: put nan here
        else:
            mask = (x != 0.0)
            y = zeros_like(asfarray(x))
            y[mask] = x[mask] ** (self.alpha)
        return y
    def f_inv(self, x):
        if isscalar(x):
            if x != 0:
                y = x ** (self.alpha_inv)
            else:
                y = 0 # TODO: put nan here
        else:
            mask = (x != 0.0)
            y = zeros_like(asfarray(x))
            y[mask] = x[mask] ** self.alpha_inv
        return y
    def f_inv_deriv(self, x):
        if isscalar(x):
            y = self.alpha_inv * x ** self.exp_deriv
        else:
            mask = (x != 0.0)
            y = zeros_like(asfarray(x))
            y[mask] = self.alpha_inv * x ** self.exp_deriv
        return y

class AbsDistr(OpDistr):
    """Absolute value of a distribution."""
    def __init__(self, d):
        super(AbsDistr, self).__init__([d])
        self.d = d
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyAbsComposition()
    def pdf(self, x):
        if isscalar(x):
            if x < 0:
                y = 0
            else:
                y = self.d.pdf(-x) + self.d.pdf(x)
        else:
            y = zeros_like(asfarray(x))
            mask = x >= 0
            y[mask] = self.d.pdf(-x[mask]) + self.d.pdf(x[mask])
        return y
    def rand_op(self, n, cache):
        return abs(self.d.rand(n, cache))
    def __str__(self):
        return "|#{0}|".format(id(self.d))
    def getName(self):
        return "|{0}|".format(self.d.getName())

class SquareDistr(OpDistr):
    """Injective function of random variable"""
    def __init__(self, d):
        super(SquareDistr, self).__init__([d])
        self.d = d
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copySquareComposition()
    def pdf(self,x):
        if x <= 0:  # won't work for x == 0
            f = 0
        else:
            f = (self.d.pdf(-sqrt(x)) + self.d.pdf(sqrt(x))) /(2*sqrt(x))
        return f
    def rand_op(self, n, cache):
        r = self.d.rand(n, cache)
        return r * r
    def __str__(self):
        return "#{0}**2".format(id(self.d))
    def getName(self):
        return "sqr({0})".format(self.d.getName())


def sqrt(d):
    if isinstance(d, Distr):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        return d ** 0.5
    if isinstance(d, RV):
        return PowRV(d, 0.5)
    return numpy.sqrt(d)

class FuncNoninjectiveDistr(OpDistr):
    """Non-injective function of random variable only piecewise smooth functions are permitted"""
    def __init__(self, d, fname="f"):
# ====================================
#        self.intervals = []
#        self.fs = []
#        self.f_invs = []
#        self.f_inv_derivs = []
#        self.fname = "none"
# ====================================
        self.fname = fname
        self.d = d
        super(FuncNoninjectiveDistr, self).__init__([d])
    def pdf(self, x):
        return self.get_piecewise_pdf()(x)
#    def pdf(self, x):
#        # TODO repair this
#        f = self.d.pdf(self.f_inv(x)) * abs(self.f_inv_deriv(x))
#        if isscalar(x):
#            if not isfinite(f):
#                f = 0
#        else:
#            mask = isfinite(f)
#            f[~mask] = 0
#        return f
    def __str__(self):
        return "{0}(#{1})".format(self.fname, id(self.d))
    def getName(self):
        return "{0}({1})".format(self.fname, self.d.getName())
    def rand_op(self, n, cache):
        X = self.d.rand(n, cache)
        Y = zeros_like(X)
        for i in range(len(self.intervals)):
            mask = (self.intervals[i][0] <= X) * (X <= self.intervals[i][1])
            Y[mask] = self.fs[i](X[mask])
        return Y
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyCompositionNoninjective(self.intervals, self.fs, self.f_invs, self.f_inv_derivs, pole_at_zero=self.pole_at_zero)

class Sq2Distr(FuncNoninjectiveDistr):
    """Exponent of a random variable"""
    def __init__(self, d):
        self.intervals = [[-Inf, 0], [0, +Inf]]
        self.fs = [lambda x: x**2, lambda x: x**2]
        self.f_invs = [lambda x: x**0.5,lambda x: -x**0.5]
        self.f_inv_derivs = [lambda x: 0.5*x**(-0.5),lambda x: -0.5*x**(-0.5)]
        self.pole_at_zero = True
        super(Sq2Distr, self).__init__(d, fname = "sin")
    def is_nonneg(self):
        return True

def _pi_m_arcsin(x):
    return 2*pi-arccos(x)
def _arcsin_der1(x):
    return (1-x**2)**(-0.5)
def _arcsin_der2(x):
    return -(1-x**2)**(-0.5)
class SinDistr(FuncNoninjectiveDistr):
    """Exponent of a random variable"""
    def __init__(self, d):
        self.intervals = [[-pi/2, pi/2], [pi/2, 3*pi/2]]
        self.fs = [numpy.sin, numpy.sin]
        self.f_invs = [numpy.arcsin, _pi_m_arcsin]
        self.f_inv_derivs = [_arcsin_der1, _arcsin_der1]
        self.pole_at_zero = False
        super(SinDistr, self).__init__(d, fname = "sin")
    def is_nonneg(self):
        return True
def sin(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return SinDistr(d)
    return numpy.sin(d)

def _twopi_m_arccos(x):
    return 2*pi-arccos(x)
def _arccos_der1(x):
    return -(1-x**2)**(-0.5)
def _arccos_der2(x):
    return (1-x**2)**(-0.5)
class CosDistr(FuncNoninjectiveDistr):
    """Exponent of a random variable"""
    def __init__(self, d):
        self.intervals = [[0.0, pi], [pi, 2.0*pi]]
        self.fs = [numpy.cos, numpy.cos]
        self.f_invs = [numpy.arccos, _twopi_m_arccos]
        self.f_inv_derivs = [_arccos_der1, _arccos_der2]
        self.pole_at_zero = False
        super(CosDistr, self).__init__(d, fname = "cos")
    def is_nonneg(self):
        return True
def cos(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return CosDistr(d)
    return numpy.cos(d)

class TanDistr(FuncDistr):
    """Tangent of a random variable"""
    def __init__(self, d):
        super(TanDistr, self).__init__(d, pole_at_zero = False, fname = "tan")
    def f(self, x):
        return numpy.tan(x)
    def f_inv(self, x):
        return numpy.arctan(x)
    def f_inv_deriv(self, x):
        y = 1.0 / (1 + x**2)
        return y

def tan(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return TanDistr(d)
    return numpy.tan(d)

class DiscreteDistr(Distr):
    """Discrete distribution"""
    def __init__(self, xi=[0.0, 1.0], pi=[0.5, 0.5]):
        super(DiscreteDistr, self).__init__([])
        assert(len(xi) == len(pi))
        px = list(zip(xi, pi))
        px.sort()
        self.px = px
        self.xi = [p[0] for p in px]
        self.pi = [p[1] for p in px]
        self.cumP = cumsum(self.pi)
    def pdf(self, x):
        """it override pdf() method to obtain discrete probabilities"""
        x = asarray(x)
        yy = zeros_like(x, dtype=float)
        for i in range(len(self.pi)):
            yy[x==self.xi[i]] += float(self.pi[i])
        return yy
    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        for i in range(len(self.xi)):
            self.piecewise_pdf.addSegment(DiracSegment(self.xi[i], self.pi[i]))
        for i in range(len(self.xi)-1):
            self.piecewise_pdf.addSegment(ConstSegment(self.xi[i], self.xi[i+1], 0))
    def rand_raw(self, n):
        u = uniform(0, 1, n)
        i = searchsorted(self.cumP, u)
        i[i > len(self.xi)] = len(self.xi)
        return array(self.xi)[i]
    def __str__(self):
        pstr = ", ".join("{0}:{1}".format(x, p) for x, p in self.px)
        return "Discrete({0})#{1}".format(pstr, self.id())
    def getName(self):
        return "Di({0})".format(len(self.xi))

class SignDistr(DiscreteDistr):
    def __init__(self, d):
        self.d = d
        prPlus = float(self.d.ccdf_value(0))
        diracZero = self.d.get_piecewise_pdf().getDirac(0)
        if diracZero is None:
            prZero = 0
        else:
            prZero = diracZero.f
        prMinus = float(self.d.cdf(0)) - prZero
        if prZero > 0:
            xi = [-1,0,1]; pi=[prMinus, prZero, prPlus]
        else:
            xi = [-1,1]; pi=[prMinus, prPlus]
        super(SignDistr, self).__init__(xi, pi)
        self.parents = [d]
    def __str__(self):
        return "sign({0})".format(id(self.d))
    def getName(self):
        return "sign({0})".format(self.d.getName())



class SumDistr(SumRV, OpDistr):
    """Sum of distributions."""
    def __init__(self, d1, d2):
        super(SumDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 + r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = conv(self.d1.get_piecewise_pdf(), self.d2.get_piecewise_pdf())
class SubDistr(SubRV, OpDistr):
    """Difference of distributions."""
    def __init__(self, d1, d2):
        super(SubDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 - r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = conv(self.d1.get_piecewise_pdf(),
                                  self.d2.get_piecewise_pdf().copyShiftedAndScaled(scale = -1))

class MulDistr(MulRV, OpDistr):
    def __init__(self, d1, d2):
        super(MulDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 * r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convprod(self.d1.get_piecewise_pdf(),
                                      self.d2.get_piecewise_pdf())

class DivDistr(DivRV, OpDistr):
    def __init__(self, d1, d2):
        super(DivDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 / r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convdiv(self.d1.get_piecewise_pdf(),
                                     self.d2.get_piecewise_pdf())
class MinDistr(MinRV, OpDistr):
    def __init__(self, d1, d2):
        super(MinDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return minimum(r1, r2)
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convmin(self.d1.get_piecewise_pdf(),
                                     self.d2.get_piecewise_pdf())
class MaxDistr(MaxRV, OpDistr):
    def __init__(self, d1, d2):
        super(MaxDistr, self).__init__(d1, d2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        #print "----", r1,r2
        return maximum(r1, r2)
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convmax(self.d1.get_piecewise_pdf(),
                                     self.d2.get_piecewise_pdf())
_builtin_min = min
def min(*args):
    if len(args) != 2:
        return _builtin_min(*args)
    d1 = args[0]
    d2 = args[1]
    if isinstance(d1, Distr) and isinstance(d2, Distr):
        return MinDistr(d1, d2)
    elif isinstance(d1, Distr) and isinstance(d2, numbers.Real):
        return MinDistr(d1, ConstDistr(d2))
    elif isinstance(d1, numbers.Real) and isinstance(d2, Distr):
        return MinDistr(ConstDistr(d1), d2)
    elif isinstance(d1, Distr) or isinstance(d2, Distr):
        raise TypeError("unorderable types: {}() < {}()".format(type(d1).__name__, type(d2).__name__))
    elif isinstance(d1, RV) and isinstance(d2, RV):
        return MinRV(d1, d2)
    else:
        return _builtin_min(*args)
_builtin_max = max
def max(*args):
    if len(args) != 2:
        return _builtin_max(*args)
    d1 = args[0]
    d2 = args[1]
    if isinstance(d1, Distr) and isinstance(d2, Distr):
        return MaxDistr(d1, d2)
    elif isinstance(d1, Distr) and isinstance(d2, numbers.Real):
        return MaxDistr(d1, ConstDistr(d2))
    elif isinstance(d1, numbers.Real) and isinstance(d2, Distr):
        return MaxDistr(ConstDistr(d1), d2)
    elif isinstance(d1, Distr) or isinstance(d2, Distr):
        raise TypeError("unorderable types: {}() < {}()".format(type(d1).__name__, type(d2).__name__))
    elif isinstance(d1, RV) and isinstance(d2, RV):
        return MaxRV(d1, d2)
    else:
        return _builtin_max(*args)

class ConstDistr(DiscreteDistr):
    def __init__(self, c = 0.0,  p=1.0, **kwargs):
        super(ConstDistr, self).__init__([c], [p], **kwargs)
        self.c = c
    def rand_raw(self, n = None):
        r = zeros(n)
        r.fill(self.c)
        return r
    def __str__(self):
        return str(self.c)
    def getName(self):
        return str(self.c)

class CondGtDistr(Distr):
    def __init__(self, d, L=None, **kwargs):
        self.L = L
        self.d = d
        super(CondGtDistr, self).__init__([], **kwargs)
    def init_piecewise_pdf(self):
        Z = MaxDistr(ConstDistr(self.L), self.d)
        # Do poprawki zostaja tu diraki o wys.=0, tak samo Lt
        #print Z.get_piecewise_pdf(), self.L
        while(Z.get_piecewise_pdf().segments[0].a<self.L):
            diracB = Z.get_piecewise_pdf().segments.pop(0)
        diracB = Z.get_piecewise_pdf().segments.pop(0)
        #print Z.get_piecewise_pdf(), diracB
        self.piecewise_pdf = (Z * DiscreteDistr(xi=[1.0], pi=[1.0/(1-diracB.f)])).get_piecewise_pdf()
    def __str__(self):
        return "{0} | X>{1}".format(self.d, self.L)
    def getName(self):
        return "{0} | X>{1}".format(self.d.getName(), self.L)
    def rand_raw(self, n):
        tab=array([])
        k=0
        while k<n:
            x =self.d.rand(n)
            tab = concatenate((tab,x[x>self.L]))
            k = len(tab)
        return tab[0:n] #self.rand_invcdf(n)
    def range(self):
        return self.L, self.d.range()[1]

class CondLtDistr(Distr):
    def __init__(self, d, U=None, **kwargs):
        self.U = U
        self.d = d
        super(CondLtDistr, self).__init__([], **kwargs)
    def init_piecewise_pdf(self):
        Z = MinDistr(ConstDistr(self.U), self.d)
        diracB = Z.get_piecewise_pdf().segments.pop(-1)
        self.piecewise_pdf = (Z * DiscreteDistr(xi=[1.0], pi=[1.0/(1-diracB.f)])).get_piecewise_pdf()
    def __str__(self):
        return "{0} | X<{1}".format(self.d, self.U)
    def getName(self):
        return "{0} | X<{1}".format(self.d.getName(), self.U)
    def rand_raw(self, n):
        return self.rand_invcdf(n)
    def range(self):
        return self.d.range()[0], self.U

class CondBetweenDistr(Distr):
    def __init__(self, d, L=None, U=None, **kwargs):
        self.L = L
        self.U = U
        self.d = d
        super(CondBetweenDistr, self).__init__([], **kwargs)
    def init_piecewise_pdf(self):
        #Z = MaxDistr(ConstDistr(self.L), self.d)
        #Z = MinDistr(Z, ConstDistr(self.U))
        #Z = min(max(self.d, self.L), self.U)
        Z = max(min(self.d, self.U), self.L)
        diracL = Z.get_piecewise_pdf().segments.pop(0)
        diracU = Z.get_piecewise_pdf().segments.pop(-1)
        pmass = 1. - (diracL.f + diracU.f)
        self.piecewise_pdf = (Z * DiscreteDistr(xi=[1.0], pi=[1./pmass])).get_piecewise_pdf()

    def __str__(self):
        return "{0} | {1}<X<{2}".format(self.d, self.L, self.U)
    def getName(self):
        return "{0} | {1}<X<{2}".format(self.d.getName(), self.L, self.U)
    def rand_raw(self, n):
        return self.rand_invcdf(n)
    def range(self):
        return self.L, self.U

class Condition(object):
    pass
class Gt(Condition):
    def __init__(self, L):
        super(Gt, self).__init__()
        self.L = L
class Lt(Condition):
    def __init__(self, U):
        super(Lt, self).__init__()
        self.U = U
class Between(Condition):
    def __init__(self, L, U):
        super(Between, self).__init__()
        self.L = L
        self.U = U

import pylab
from pylab import plot, subplot, xlim, ylim, show, figure
def demo_distr(d,
               theoretical = None,
               err_plot = True,
               test_mode = False,
               tails = False,
               histogram = False,
               summary = True,
               xmin = None, xmax = None,
               ymin = None, ymax = None,
               title = None,
               n_points = 1000,
               hist_points = 1000000,
               hist_bins = 50,
               log_scale = True):
    """Plot or test a distribution, error etc."""
    if title is None:
        title = d.getName()
    if err_plot and theoretical is None:
        histogram = True
    if theoretical is not None:
        # compute error against theoretical
        f = d.get_piecewise_pdf()
        X = f.getPiecewiseSpace(numberOfPoints = n_points, xmin = xmin, xmax = xmax)
        #Yf = d(X)  # this should really be used...
        Yf = f(X)
        Yt = theoretical(X)
        if summary or test_mode:
            maxabserr = max(abs(Yf - Yt))
            relerr = abs(Yf - Yt)/Yt
            relerr[Yt == 0] = 0
            maxrelerr = max(relerr)
    if summary or test_mode:
        f = d.get_piecewise_pdf()
        I = f.integrate()
    if not test_mode:
        if theoretical:
            pylab.subplot(211)
            plot(X, Yt, color='c', linewidth=4)
        d.plot(numberOfPoints = n_points, xmin = xmin, xmax = xmax, color='k')
        if histogram:
            d.hist(n = hist_points, xmin = xmin, xmax = xmax, bins = hist_bins)
        if xmin is not None:
            xlim(xmin = xmin)
        if xmax is not None:
            xlim(xmax = xmax)
        if ymin is not None:
            ylim(ymin = ymin)
        if ymax is not None:
            ylim(ymax = ymax)
        if theoretical:
            pylab.subplot(212)
            abse = abs(Yf - Yt)
            if isinstance(theoretical, Distr):
                r = f - theoretical.get_piecewise_pdf()
                log_scale = False
                r.plot(numberOfPoints = n_points, xmin = xmin, xmax = xmax, color='k')
                ss = d.summary_map()
                sd = theoretical.summary_map()
                print("============= summary =============")
                print(" ", d.getName())
                for i in ['mean', 'std', 'var', 'median', 'entropy', 'medianad', 'iqrange(0.025)',  'ci(0.05)', 'range', 'int_err']:
                    if i in ss:
                        try:
                            if i=='int_err':
                                r = abs(ss[i]-0)
                            else:
                                r = abs(ss[i]-sd[i])
                            if not r==r:
                                r = 0.0
                            print('{0:{align}20}'.format(i, align = '>'), "=", '{0:{align}24}'.format(repr(ss[i]), align = '>'), " +/-", '%1.3g' % r)
                        except Exception as e:
                            pass
            else:
                pylab.plot(X, abse, color='k')
                d.summary()
            pylab.ylabel("abs. error")
            if max(abse) == 0:
                log_scale = False
            if log_scale:
                pylab.gca().set_yscale("log")
        if title is not None:
            pylab.suptitle(title)
    if summary and not theoretical:
        d.summary()
    if summary and theoretical:
        print("max. abs. error", maxabserr)
        print("max. rel. error", maxrelerr)
    #show()
