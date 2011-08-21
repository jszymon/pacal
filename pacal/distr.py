"""Base distribution classes.  Operations on distributions."""

import numbers
from functools import partial

import numpy
from numpy import array, zeros_like, unique, concatenate, isscalar, isfinite
from numpy import sqrt, pi, arctan, tan, asfarray
from numpy.random import uniform
from numpy import minimum, maximum
from numpy import hstack
from numpy import histogram

from utils import Inf

from pylab import bar

import traceback

import params
from indeparith import conv, convprod, convdiv, convmin, convmax

class Distr(object):
    def __init__(self, parents = [], indep = True):
        # indep = True means the distribution is treated as
        # independent from all others.  For examples this results in
        # X+X != 2X.  This currently only affects random number
        # generation and histograms.  This default will likely change
        # in the future.
        self.parents = parents
        self.indep = indep
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
                    print "Warning: arguments treated as independent"
                    break
                anc.update(panc)
    def __str__(self):
        return "Distr"
    def getName(self):
        """return a string representation of PDF."""
        return "D"
    def getAncestorIDs(self):
        """Get ID's of all ancestors"""
        anc = set()
        for p in self.parents:
            anc.update(p.getAncestorIDs())
        anc.add(id(self))
        return anc
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
        return self.piecewise_cdf
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
        raise NotImplemented()
    def get_piecewise_invcdf(self, use_interpolated=True):
        """return, CDF function, as CumulativePiecewiseFunction object"""
        invcdf  = self.get_piecewise_cdf().invfun(use_interpolated=use_interpolated)
        return invcdf

    def pdf(self,x):
        return self.get_piecewise_pdf()(x)
    def cdf(self,x):
        """Cumulative piecewise function."""
        return self.get_piecewise_cdf()(x)
    def ccdf(self,x):
        """Complementary cumulative piecewise function.
        Not implemented yet. """
        pass
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
    def range(self):
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
    def tailexp(self):
        """Check whether distribution is positive definite."""
        return self.get_piecewise_pdf().tailexp()
    def summary_map(self):
        r = {}
        r['mean'] = self.mean()
        r['std'] = self.std()
        r['var'] = self.var()
        r['entropy'] = self.entropy()
        r['range'] = self.get_piecewise_pdf().range()
        r['int_err'] = 1-self.get_piecewise_pdf().integrate()
        r['tailexp'] = self.tailexp()
        #r['interp_errs'] = self.getInterpErrors()
        try:
            r['median'] = self.median()
            r['iqrange(0.025)'] = self.iqrange()
            r['medianad'] = self.medianad()  
            r['ci(0.05)'] = self.ci()       
        except Exception, e:           
            traceback.print_exc() 
        return r
    def summary(self):
        """Summary statistics for a given distribution."""
        print "============= summary ============="
        #print self.get_piecewise_pdf()
        summ = self.summary_map()
        print " ", self.getName()
        for i in ['mean', 'std', 'var','entropy', 'tailexp', 'median', 'medianad', 'iqrange(0.025)',  'range', 'ci(0.05)', 'int_err']:
            if summ.has_key(i): 
                print '{0:{align}20}'.format(i, align = '>'), " = ", summ[i]       

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
        if id(self) not in cache:
            cache[id(self)] = self.rand_raw(n)
        return cache[id(self)]
    def plot(self, color='k', *args, **kvargs):
        """Plot of PDF.
        
        Keyword arguments:
        xmin -- minimum x range
        xmax -- maximum x range        
        other of pylab/plot **kvargs  
        """
        self.get_piecewise_pdf().plot(color=color, *args, **kvargs)      
    def hist(self, n = 1000000, xmin = None, xmax = None, bins = 50, max_samp = None, **kwargs):
        """Histogram of PDF.
        
        Keyword arguments:
        n -- number of pints
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
            bar(b, float(c) * w, width = width, alpha = 0.25, **kwargs)
    
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
        raise NotImplemented()
    def __radd__(self, d):
        """Overload sum with real number: distribution of X+r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, shift = d)
        raise NotImplemented()
    def __sub__(self, d):
        """Overload subtraction: distribution of X-Y."""
        if isinstance(d, Distr):
            return SubDistr(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, shift = -d)
        raise NotImplemented()
    def __rsub__(self, d):
        """Overload subtraction with real number: distribution of X-r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, scale = -1, shift = d)
        raise NotImplemented()
    def __mul__(self, d):
        """Overload multiplication: distribution of X*Y."""
        if isinstance(d, Distr):
            return MulDistr(self, d)
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledDistr(self, scale = d)
        raise NotImplemented()
    def __rmul__(self, d):
        """Overload multiplication by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledDistr(self, scale = d)
        raise NotImplemented()
    def __div__(self, d):
        """Overload division: distribution of X*r."""
        if isinstance(d, Distr):
            return DivDistr(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledDistr(self, scale = 1.0 / d)
        raise NotImplemented()
    def __rdiv__(self, d):
        """Overload division by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            #return FuncDistr(self, lambda x: d/x, lambda x: d/x, lambda x: d/x**2)
            return d * InvDistr(self)
        raise NotImplemented()
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
        raise NotImplemented()
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
        raise NotImplemented()

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
        if id(self) not in cache:
            cache[id(self)] = self.rand_op(n, cache)
        return cache[id(self)]

class FuncDistr(OpDistr):
    """Injective function of random variable"""
    def __init__(self, d, f, f_inv, f_inv_deriv, pole_at_zero = False, fname = "f"):
        super(FuncDistr, self).__init__([d])
        self.d = d
        self.f = f
        self.f_inv = f_inv
        self.f_inv_deriv = f_inv_deriv
        self.fname = fname
        self.pole_at_zero = pole_at_zero 
    def pdf(self, x):
        f = self.d.pdf(self.f_inv(x)) * abs(self.f_inv_deriv(x))
        if isscalar(x):
            if not isfinite(f):
                f = 0
        else:
            mask = isfinite(f)
            f[~mask] = 0
        return f
    def __str__(self):
        return "{0}(#{1})".format(self.fname, id(self.d))
    def getName(self):
        return "{0}({1})".format(self.fname, self.d.getName())
    def rand_op(self, n, cache):
        return self.f(self.d.rand(n, cache))
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyComposition(self.f, self.f_inv, self.f_inv_deriv, pole_at_zero = self.pole_at_zero)

class ShiftedScaledDistr(OpDistr):
    def __init__(self, d, shift = 0, scale = 1):
        assert(scale != 0)
        super(ShiftedScaledDistr, self).__init__([d])
        self.d = d
        self.shift = shift
        self.scale = scale
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
    def __str__(self):
        if self.shift == 0 and self.scale == 1:
            return str(id(self.d))
        elif self.shift == 0:
            return "{0}*#{1}".format(self.scale, id(self.d))
        elif self.scale == 1:
            return "#{0}{1:+}".format(id(self.d), self.shift)
        else:
            return "#{0}*{1}{2:+}".format(id(self.d), self.scale, self.shift)
    def getName(self):
        if self.shift == 0 and self.scale == 1:
            return self.d.getName()
        else:
            d_name = _wrapped_name(self.d)
            if self.shift == 0:
                return "{0}*{1}".format(self.scale, d_name)
            elif self.scale == 1:
                return "{0}{1:+}".format(d_name, self.shift)
            else:
                return "{2}*{0}+{1}".format(d_name, self.shift, self.scale)

class ExpDistr(FuncDistr):
    """Exponent of a random variable"""
    def __init__(self, d):
        super(ExpDistr, self).__init__(d, numpy.exp, numpy.log,
                                       lambda x: 1.0/abs(x), pole_at_zero = True, fname = "exp")
    def is_nonneg(self):
        return True
def exp(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return ExpDistr(d)
    return numpy.exp(d)
class LogDistr(FuncDistr):
    """Natural logarithm of a random variable"""
    def __init__(self, d):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        super(LogDistr, self).__init__(d, numpy.log, numpy.exp,
                                       numpy.exp, pole_at_zero= True, fname = "log")
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copyLogComposition(self.f, self.f_inv, self.f_inv_deriv, pole_at_zero = self.pole_at_zero)
    
def log(d):
    """Overload the log function."""
    if isinstance(d, Distr):
        return LogDistr(d)
    return numpy.log(d)

def sign(d):
    """Overload sign: distribution of sign(X)."""
    if isinstance(d, Distr):
        return SignDistr(d)
    return numpy.sign(d)

class AtanDistr(FuncDistr):
    """Arcus tangent of a random variable"""
    def __init__(self, d):
        super(AtanDistr, self).__init__(d, numpy.arctan, self.f_inv,
                                        self.f_inv_deriv, pole_at_zero= False, fname ="atan")
    @staticmethod
    def f_inv(x):
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
    @staticmethod
    def f_inv_deriv(x):
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
    return numpy.arctan(d)

class InvDistr(OpDistr):
    """Inverse of random variable."""
    def __init__(self, d):
        super(InvDistr, self).__init__([d])
        self.d = d
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
    def __str__(self):
        return "1/#{0}".format(id(self.d))    
    def getName(self):
        d_name = self.d.getName()
        if isinstance(self.d, OpDistr) and not isinstance(self.d, FuncDistr):
            d_name = "(" + d_name + ")"
        return "(1/{0})".format(d_name)
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

class PowDistr(FuncDistr):
    """Inverse of random variable."""
    def __init__(self, d, alpha = 1):
        super(PowDistr, self).__init__([d],self.f_, self.f_inv, self.f_inv_deriv, pole_at_zero = alpha > 1, fname="pow")
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
    def rand_op(self, n, cache):
        return self.d.rand(n, cache) ** self.alpha
    def __str__(self):
        return "#{0}^{1}".format(id(self.d1), self.alpha)
    def getName(self):
        return "{0}^{1}".format(_wrapped_name(self.d), self.alpha)    
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

class SignDistr(Distr):
    def __init__(self, d):
        self.d = d
        super(SignDistr, self).__init__(d)
    def init_piecewise_pdf(self):
        prPlus = float(self.d.ccdf_value(0))
        diracZero = self.d.get_piecewise_pdf().getDirac(0)
        if diracZero is None:
            prZero = 0
        else:
            prZero = diracZero.f
        prMinus = float(self.d.cdf(0)) - prZero        
        print "=======", [prMinus, prZero, prPlus] 
        #Z.get_piecewise_pdf().segments.remove(diracB)
        if prZero > 0:
            self.piecewise_pdf = (DiscreteDistr(xi = [-1,0,1], pi=[prMinus, prZero, prPlus])).get_piecewise_pdf()
        else:
            self.piecewise_pdf = (DiscreteDistr(xi = [-1,1], pi=[prMinus, prPlus])).get_piecewise_pdf()
    def __str__(self):
        return "sign({0})".format(id(self.d))
    def getName(self):
        return "sign({0})".format(self.d.getName())

class SquareDistr(OpDistr):
    """Injective function of random variable"""
    def __init__(self, d):
        super(SquareDistr, self).__init__()
        self.d = d
    def init_piecewise_pdf(self):
        self.piecewise_pdf = self.d.get_piecewise_pdf().copySquareComposition()
    #def pdf(self,x):
    #    if x <= 0:  # won't work for x == 0
    #        f = 0
    #    else:
    #        f = (self.d.pdf(-sqrt(x)) + self.d.pdf(sqrt(x))) /(2*sqrt(x))
    #    return f
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
    return numpy.sqrt(d)

class SumDistr(OpDistr):
    """Sum of distributions."""
    def __init__(self, d1, d2):
        super(SumDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}+#{1}".format(id(self.d1), id(self.d2))
    def getName(self):
        return "{0}+{1}".format(self.d1.getName(), self.d2.getName())
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 + r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = conv(self.d1.get_piecewise_pdf(), self.d2.get_piecewise_pdf())
class SubDistr(OpDistr):
    """Difference of distributions."""
    def __init__(self, d1, d2):
        super(SubDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}-#{1}".format(id(self.d1), id(self.d2))
    def getName(self):
        n2 = _wrapped_name(self.d2, incl_classes = [SumDistr])
        return "{0}-{1}".format(self.d1.getName(), n2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 - r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = conv(self.d1.get_piecewise_pdf(),
                                  self.d2.get_piecewise_pdf().copyShiftedAndScaled(scale = -1))

class MulDistr(OpDistr):
    def __init__(self, d1, d2):
        super(MulDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}*#{1}".format(id(self.d1), id(self.d2))
    def getName(self):
        n1 = _wrapped_name(self.d1, incl_classes = [SumDistr, SubDistr, ShiftedScaledDistr])
        n2 = _wrapped_name(self.d2, incl_classes = [SumDistr, SubDistr, ShiftedScaledDistr])
        return "{0}*{1}".format(n1, n2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 * r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convprod(self.d1.get_piecewise_pdf(),
                                      self.d2.get_piecewise_pdf())
    
class DivDistr(OpDistr):
    def __init__(self, d1, d2):
        super(DivDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}/#{1}".format(id(self.d1), id(self.d2))
    def getName(self):
        n1 = _wrapped_name(self.d1)
        n2 = _wrapped_name(self.d2)
        return "{0}/{1}".format(n1, n2)
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return r1 / r2
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convdiv(self.d1.get_piecewise_pdf(),
                                     self.d2.get_piecewise_pdf())
class MinDistr(OpDistr):
    def __init__(self, d1, d2):
        super(MinDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "min(#{0}, #{1})".format(id(self.d1), id(self.d2))
    def getName(self):
        return "min({0}, {1})".format(self.d1.getName(), self.d2.getName())
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
        return minimum(r1, r2)
    def init_piecewise_pdf(self):
        self.piecewise_pdf = convmin(self.d1.get_piecewise_pdf(),
                                     self.d2.get_piecewise_pdf())
class MaxDistr(OpDistr):
    def __init__(self, d1, d2):
        super(MaxDistr, self).__init__([d1, d2])
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "max(#{0}, #{1})".format(id(self.d1), id(self.d2))
    def getName(self):
        return "max({0}, {1})".format(self.d1.getName(), self.d2.getName())
    def rand_op(self, n, cache):
        r1 = self.d1.rand(n, cache)
        r2 = self.d2.rand(n, cache)
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
    elif isinstance(d1, Distr) or isinstance(d2, Distr):
        raise NotImplemented()
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
    elif isinstance(d1, Distr) or isinstance(d2, Distr):
        raise NotImplemented()
    else:
        return _builtin_max(*args)

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
                print "============= summary ============="
                print " ", d.getName()
                for i in ['mean', 'std', 'var', 'median', 'medianad', 'iqrange(0.025)',  'ci(0.05)', 'range', 'int_err']:
                    if ss.has_key(i): 
                        try:
                            if i=='int_err':
                                r = abs(ss[i]-0)
                            else:
                                r = abs(ss[i]-sd[i])
                            if not r==r:
                                r = 0.0                           
                            print '{0:{align}20}'.format(i, align = '>'), "=", '{0:{align}24}'.format(repr(ss[i]), align = '>'), " +/-", '%1.3g' % r    
                        except Exception, e:  
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
        print "max. abs. error", maxabserr
        print "max. rel. error", maxrelerr
    #show()
