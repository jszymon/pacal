"""PiecewiseFunction for piecewise functions."""

from __future__ import print_function

import bisect
import numbers
import operator
from functools import partial

from .integration import *
from .interpolation import *
from .utils import epsunique, estimateDegreeOfPole, testPole, findinv, findinv_minf, findinv_pinf, estimateTailExponent, is_instance_method

from . import params

from numpy import asfarray
from numpy import linspace, multiply, add, divide, size
from numpy import unique, union1d, isnan, isscalar, diff, size
from numpy import inf, Inf, NaN, sign, isinf, isfinite, exp, isneginf, isposinf
from numpy import logspace, sqrt, minimum, maximum, pi, mean, log10
from numpy import append, nan_to_num, select
from numpy.random import uniform
from numpy import argmax, argmin
import numpy
from pylab import plot, semilogx, xlabel, ylabel, axis, loglog, figure, subplot

try:
    from scipy.optimize.optimize import fminbound
    have_Scipy_optimize = True
except ImportError:
    have_Scipy_optimize = False



def prob_comp_fun(f, ginv, ginvderiv, x):
    """Helper function for composition"""
    if isscalar(x):
        try:
            y = f(ginv(x)) * abs(ginvderiv(x))
        except ArithmeticError:
            y = 0
    else:
        y = _safe_call(f(ginv(x)) * abs(ginvderiv(x)))
    return y
def call_segint(seg, y0, x):
    if isscalar(x):
        return y0 + seg.segIntegral(x)
    else:
        return y0 + array([seg.segIntegral(xi) for xi in x])
def call_segint2(seg, y0, x):
    if isscalar(x):
        return y0 - seg.segIntegral(x)
    else:
        return y0 - array([seg.segIntegral(xi) for xi in x])

def _shift_and_scale(f, one_over_scale, shift, x):
    return abs(one_over_scale) * f((x - shift) * one_over_scale)

def _op_rsub(x, y):
    return y - x
def _op_rdiv(x, y):
    return y / x
def _op_square(x):
    return x * x
def _op_sqrt(x):
    return x ** 0.5
def _op_half_over_sqrt(x):
    return 0.5 / sqrt(x)
def _op_one_over_x(x):
    if isscalar(x):
        if x != 0:
            y = 1.0 / x
        else:
            y = Inf
    else:
        mask = (x != 0.0)
        y = zeros_like(asfarray(x))
        y[mask] = 1.0 / x[mask]
    return y
def _op_one_over_xsqaure(x):
    if isscalar(x):
        if x != 0:
            y = 1.0 / x**2
        else:
            y = Inf
    else:
        mask = (x != 0.0)
        y = zeros_like(asfarray(x))
        y[mask] = 1.0 / (x[mask])**2
    return y

def _prob_composition(f, ginv, ginvderiv, x):
    return f(ginv(x)) * abs(ginvderiv(x))
def _prob_composition2(f, ginv, ginvderiv, x):
    return f(-ginv(x)) * abs(ginvderiv(x))
def _prob_composition_safe(f, ginv, ginvderiv, x):
    return _safe_call(f(ginv(x)) * abs(ginvderiv(x)))

def _call_f(obj, x):
    return obj.f(x)
def wrap_f(f):
    if params.general.parallel:
        if is_instance_method(f):
            return partial(_call_f, f.__self__)
        return f
    return f
def _safe_fun_on_f_call(seg, fun, x):
    return nan_to_num(fun(seg.f(x)))
def _segfun_scalar_op(seg, op, r, x):
    return op(seg.f(x), r)
def _scalar_segfun_op(seg, op, r, x):
    return op(r, seg.f(x))
def _segfun_segfun_op(seg1, seg2, op, x):
    return op(seg1.f(x), seg2.f(x))


class Segment(object):
    """Segment of piecewise continuous function,
    default on finite interval [a, b].
    """
    def __init__(self, a, b, f, safe_a = None, safe_b = None):
        self.a = float(a)
        self.b = float(b)
        if f is not None:
            self.f = f
        # useful e.g. for singularities at ends
        if safe_a is None:
            safe_a = a
        if safe_b is None:
            safe_b = b
        self.safe_a = float(safe_a) # TODO: why do we need floats here?
        self.safe_b = float(safe_b)

    def __str__(self):
        return "{0}, [{1}, {2}]".format(self.__class__.__name__, self.a, self.b)
    def __repr__(self):
        return self.__str__()
    def __call__(self, x):
        if isscalar(x):
            if self.a <= x <= self.b:
                return self.f(x)
            else:
                return 0
        y = zeros_like(x)
        ind = where((x>=self.a) & (x<=self.b))
        if len(ind)>0:
            y[ind] = self.f(x[ind])
        return y

    def integrate(self, a = None, b = None):
        """definite integral over interval (c, d) \cub (a, b) """
        if a==None or a<self.a :
            a=self.a
        if b==None or b>self.b:
            b=self.b
        #i,e = integrate_fejer2(self, self.a, self.b, debug_plot = False)
        i,e = integrate_fejer2(self, a, b, debug_plot = False)
        return i
    def cumint(self, y0 = 0.0):
        """indefinite integral over interval (a, x)"""
        return Segment(self.a , self.b, partial(call_segint, self, y0))
    def segIntegral(self, x):
        if isscalar(x):
            return self.integrate(self.a, x)
        else:
            return array([self.integrate(self.a, xi) for xi in x])
    def diff(self):
        return self.toInterpolatedSegment().diff()
    def roots(self):
        return self.toInterpolatedSegment().roots()
    def trim(self):
        return self

    def toInterpolatedSegment(self, left_pole = False, NoL = False, right_pole = False, NoR = False):
        if left_pole or right_pole:
            iseg = InterpolatedSegmentWithPole(self.a, self.b, self.f, left_pole = left_pole)
        elif NoL and NoR:
            iseg = InterpolatedSegment(self.a, self.b, ChebyshevInterpolator1(self.f, self.a, self.b))
        elif NoL:
            iseg = InterpolatedSegment(self.a, self.b, ChebyshevInterpolatorNoL(self.f, self.a, self.b, par = params.interpolation_finite))
        elif NoR:
            iseg = InterpolatedSegment(self.a, self.b, ChebyshevInterpolatorNoR(self.f, self.a, self.b, par = params.interpolation_finite))
        elif params.interpolation.use_cheb_2nd:
            iseg = InterpolatedSegment(self.a, self.b, ChebyshevInterpolator(self.f, self.a, self.b, par = params.interpolation_finite))
        else:
            iseg = InterpolatedSegment(self.a, self.b, ChebyshevInterpolator1(self.f, self.a, self.b, par = params.interpolation_finite))
        return iseg

    def findLeftpoint(self):
        return self.a
    def findRightpoint(self):
        return self.b
    def getSegmentSpace(self,
             xmin = None,
             xmax = None,
             numberOfPoints = None):
        if numberOfPoints is None:
            numberOfPoints = params.segments.plot.numberOfPoints
        leftRightEpsilon = params.segments.plot.leftRightEpsilon
        if (xmin == None):
            xmin = self.findLeftpoint()
        else:
            xmin = max(xmin, self.a)
        if (xmax == None):
            xmax = self.findRightpoint()
        else:
            xmax = min(xmax, self.b)
        if xmin>=xmax:
            return array([])
        if xmin==0:
            xmin = xmin + leftRightEpsilon
        if xmax==0:
            xmax = xmax - leftRightEpsilon
        if xmin == 0.0 or xmax/xmin>1e2:
            xi = logspace(log10(abs(xmin)), log10(abs(xmax)), numberOfPoints)
        elif xmax == 0.0 or xmin/xmax>1e2:
            xi = -logspace( log10(abs(xmin)), log10(abs(xmax)), numberOfPoints)
        else:
            xi = linspace(xmin, xmax, numberOfPoints)
        return xi
    def plot(self,
             xmin = None,
             xmax = None,
             show_nodes = True,
             show_segments = True,
             numberOfPoints = params.segments.plot.numberOfPoints, **args):
        leftRightEpsilon = params.segments.plot.leftRightEpsilon

        if (xmin == None):
            xmin = self.findLeftpoint()
        else:
            xmin = max(xmin, self.a)
        if (xmax == None):
            xmax = self.findRightpoint()
        else:
            xmax = min(xmax, self.b)
        if xmin>=xmax:
            return
        if xmin==0:
            xmin = xmin + leftRightEpsilon
        if xmax==0:
            xmax = xmax - leftRightEpsilon
        if (xmin == 0.0 or xmax/xmin>1e2) and self.hasLeftPole():
            xi = logspace(log10(abs(xmin)), log10(abs(xmax)), numberOfPoints)
        elif (xmax == 0.0 or xmin/xmax>1e2) and self.hasRightPole():
            xi = -logspace( log10(abs(xmin)), log10(abs(xmax)), numberOfPoints)
        else:
            xi = linspace(xmin, xmax, numberOfPoints)
        yi = self.f(xi)
        plot(xi, yi,'-', **args)
        # show segment
        if not self.isMorPInf() and show_segments:
            plot([xmin, xmin], [0, yi[0]], 'c--', linewidth=1)
            plot([xmax, xmax], [0, yi[-1]], 'c--', linewidth=1)
        # show vertical asympthote
        if self.hasLeftPole() and xmin == self.a:
            plot([xmin, xmin], [0, yi[0]], 'k-.', linewidth=1)
        elif self.hasRightPole() and xmax == self.b:
            plot([xmax, xmax], [0, yi[-1]], 'k-.', linewidth=1)

    def semilogx(self, **args):
        numberOfPoints = params.segments.plot.numberOfPoints
        xmin = self.a
        xmax = self.b
        if (xmin == -Inf):
            xmin = self.findLeftpoint()
        if (xmax == Inf):
            xmax = self.findRightpoint()
        #xi = logspace(log10(xmin), log10(xmax), numberOfPoints)
        #xi = linspace(xmin, xmax, numberOfPoints)
        xi = linspace(xmin + 1e-8, xmax - 1e-8, numberOfPoints)
        #yi = self.__call__(xi)
        yi = self.f(xi)
        semilogx(xi, yi,'-', **args)
        if not self.isMorPInf():
            semilogx([xmin, xmin], [0, yi[0]], 'c--', linewidth=1)
            semilogx([xmax, xmax], [0, yi[-1]], 'c--', linewidth=1)
    def shiftAndScale(self, shift, scale):
        """Scale and shift a segment: replace scale*f(x)+shift
        """
        if scale is 1:
            _1_scale = 1
        else:
            _1_scale = 1.0 / scale
        if scale > 0:
            a = self.a
            b = self.b
        else:
            a = self.b
            b = self.a
        return Segment(a * scale + shift, b * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift))
    def probComposition(self, g, ginv, ginvderiv, pole_at_zero = False):
        """It produce probabilistic composition g o f for a given distribution f
        """
        if self.isDirac():
            return DiracSegment(g(self.a), self.f)
        fun = partial(prob_comp_fun, self.f, ginv, ginvderiv)
        #print fun.func, fun.args
        if (isinf(g(array([self.a])))):
        #if (self.a==0):
            if  g(self.b)<0:
                return MInfSegment(g(self.b), fun)
            else:
                return PInfSegment(g(self.b), fun)
        if (isinf(g(array([self.b])))):
        #elif (self.b==0):
            if g(self.a)>0:
                return PInfSegment(g(self.a), fun)
            else:
                return MInfSegment(g(self.a), fun)
        else:
            if g(self.a) < g(self.b):
                if g(self.a)==0.0 and pole_at_zero:
                    return SegmentWithPole(g(self.a), g(self.b), fun)
                else:
                    return Segment(g(self.a),g(self.b), fun)
            elif g(self.a) == g(self.b):
                return DiracSegment(g(self.a), self.integrate())
            else:
                if g(self.b)==0.0 and pole_at_zero:
                    return SegmentWithPole(g(self.b), g(self.a), fun, left_pole = False)
                else:
                    return Segment(g(self.b),g(self.a), fun)
    # TODO: unused
    def probInverse(self, pole_at_zero = False): # TODO unused ??
        """It produce probalilistic composition g o f for a given distribution f
        """
        g = _op_one_over_x
        ginv = _op_one_over_x
        fun = partial(_prob_composition_safe, self.f, _op_one_over_x, _op_one_over_xsqaure)
        if (self.a==0):
            if  g(self.b)<0:
                return MInfSegment(g(self.b), fun)
            else:
                return PInfSegment(g(self.b), fun)
        #if (isinf(g(self.b))):
        elif (self.b==0):
            if g(self.a)>0:
                return PInfSegment(g(self.a), fun)
            else:
                return MInfSegment(g(self.a), fun)
        else:
            if g(self.a)<=g(self.b):
                #return Segment(g(self.a)+8* finfo(float).eps,g(self.b)-4* finfo(float).eps, fun)
                if g(self.a)==0.0 and pole_at_zero:
                    return SegmentWithPole(g(self.a), g(self.b), fun)
                else:
                    return Segment(g(self.a), g(self.b), fun)

            else:
                #return Segment(g(self.b)+4* finfo(float).eps,g(self.a)-4* finfo(float).eps, fun)
                #return Segment(g(self.b), g(self.a), fun)
                if g(self.a)==0.0 and pole_at_zero:
                    return SegmentWithPole(g(self.b), g(self.a), fun)
                else:
                    return Segment(g(self.b), g(self.a), fun)

    def squareComposition(self):
        """Produce square of a random variable X^2 ~ f for a given distribution f over segment [a,b]
        we assume that:  a * b >= 0
        """
        g = _op_square
        ginv = _op_sqrt
        ginvderiv = _op_half_over_sqrt
        if self.isDirac():
            return DiracSegment(g(self.a), self.f)
        assert (self.a>=0 and self.b>0) or (self.a<0 and self.b<=0)
        if self.a>=0:
            if (isinf(self.b)):
                return PInfSegment(g(self.a), partial(_prob_composition, self.f, ginv, ginvderiv))
            else:
                if self.a>0:
                    return Segment(g(self.a), g(self.b), partial(_prob_composition, self.f, ginv, ginvderiv))
                else:
                    return SegmentWithPole(g(self.a), g(self.b), partial(_prob_composition, self.f, ginv, ginvderiv))
        else:
            if (isinf(self.a)):
                return PInfSegment(g(self.b), partial(_prob_composition2, self.f, ginv, ginvderiv))
            else:
                if self.b<0:
                    return Segment(g(self.b), g(self.a), partial(_prob_composition2, self.f, ginv, ginvderiv))
                else:
                    return SegmentWithPole(g(self.b), g(self.a), partial(_prob_composition2, self.f, ginv, ginvderiv))

    def absComposition(self):
        """It produce absolute value of random variable |X| o f for a given distribution f over segment [a,b]
        we assume that:  a * b >= 0
        """

        g = operator.abs
        ginv = operator.abs
        ginvderiv = ConstFun(1)
        if self.isDirac():
            return DiracSegment(g(self.a), self.f)
        assert (self.a>=0 and self.b>0) or (self.a<0 and self.b<=0)
        if self.a>=0:
            if (isinf(self.b)):
                return PInfSegment(g(self.a), partial(_prob_composition, self.f, ginv, ginvderiv))
            else:
                if self.a>0:
                    return Segment(g(self.a), g(self.b), partial(_prob_composition, self.f, ginv, ginvderiv))
                else:
                    return SegmentWithPole(g(self.a), g(self.b), partial(_prob_composition, self.f, ginv, ginvderiv))
        else:
            if (isinf(self.a)):
                return PInfSegment(g(self.b), partial(_prob_composition2, self.f, ginv, ginvderiv))
            else:
                if self.b<0:
                    return Segment(g(self.b), g(self.a), partial(_prob_composition2, self.f, ginv, ginvderiv))
                else:
                    return SegmentWithPole(g(self.b), g(self.a), partial(_prob_composition2, self.f, ginv, ginvderiv))


    def __lt__(self, other):
        if (self.a <= other.a and self.b < other.b) or (self.a < other.a and self.b <= other.b)  :
            return True
        else:
            return False
    def __gt__(self, other):
        if (other.a <= self.a and other.b < self.b) or (other.a < self.a and other.b <= self.b) :
            return True
        else:
            return False
    def __le__(self, other):
        if self.b <= other.a :
            return True
        else:
            return False
    def __ge__(self, other):
        if other.b <= self.a :
            return True
        else:
            return False
    def __eq__(self, other):
        if (self.a == other.a or self.b == other.b) :
            return True
        else:
            return False
    def isSegment(self):
        return True
    def isDirac(self):
        return False
    def hasLeftPole(self):
        return False
    def hasRightPole(self):
        return False
    def hasPole(self):
        return self.hasLeftPole() or self.hasRightPole()
    def isPInf(self):
        return False
    def isMInf(self):
        return False
    def isMorPInf(self):
        return self.isMInf() or self.isPInf()

class ConstFun(object):
    def __init__(self, c):
        self.c = c
    def __call__(self, x):
        if isscalar(x):
            return self.c
        return zeros_like(x) + self.c
class ConstSegment(Segment):
    """constant function over finite interval
    """
    def __init__(self, a, b, c):
        self.const = c
        super(ConstSegment, self).__init__(a, b, ConstFun(c))
    def __call__(self, x):
        return select([(x>=self.a) & (x<=self.b)], [self.const], 0)
    def integrate(self, a = None, b = None):
        """definite integral over interval (c, d) \cub (a, b) """
        if a==None or a<self.a :
            a=self.a
        if b==None or b>self.b:
            b=self.b
        return self.const*(b-a)
    #def f(self, x):
    #    return 0.0*x + self.const
class MInfSegment(Segment):
    """Segment on the range [-inf, b)."""
    def __init__(self, b, f):
        super(MInfSegment, self).__init__(-Inf, b, f)
    def toInterpolatedSegment(self):
        return MInfInterpolatedSegment(self.b,
                                   #ChebyshevInterpolator_MInf(self.f, self.b))
                                   MInfInterpolator(self.f, self.b))
    def findLeftpoint(self):
        x = self.b - 1
        fx = self.f(array([x]))
        h0 = fx
        while not ((fx <= h0 * params.segments.plot.yminEpsilon) or (abs(fx - self.f(array([x - 1.2 * abs(x-self.b)]))) <= h0 * params.segments.plot.yminEpsilon)):
            x = x - 1.2 * abs(x - self.b)
            fx = self.f(array([x]))
            if abs(x) > 1e20:
                break
        return x
    def findLeftEps(self):
        x = self.b - 1
        while (isfinite(x) and self.f(x) > 1e-16):
            x = x-1.2*abs(x-self.b)
        return x
    def integrate(self, a = None, b = None):
        """definite integral over interval (c, d) \cub (a, b) """
        if b==None or b>self.b:
            b=self.b
        if a==None or isneginf(a):
            i,e = integrate_fejer2_minf(self.f, b, exponent = params.integration_infinite.exponent)
        elif a<b:
            i,e = integrate_fejer2_minf(self.f, b, a, exponent = params.integration_infinite.exponent)
        else:
            i,e = 0,0
        return i
    def cumint(self, y0 = 0.0):
        """indefinite integral over interval (a, x)"""
        return MInfSegment(self.b, partial(call_segint, self, y0))
    def segIntegral(self, x):
        if isscalar(x):
            return self.integrate(b = x)
        else:
            return array([self.integrate(b = xi) for xi in x])
    def shiftAndScale(self, shift, scale):
        """It produce f((x - shift)/scale) for given f(x)
        """
        _1_scale = 1.0 / scale
        if scale > 0:
            return MInfSegment(self.b * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift))
        else:
            return PInfSegment(self.b * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift))
    def isMInf(self):
        return True
    def tailexp(self):
        return estimateTailExponent(self.f, pos = False)

class PInfSegment(Segment):
    """Segment = (a, inf]
    """
    def __init__(self, a, f):
        super(PInfSegment, self).__init__(a, Inf, f)
    def toInterpolatedSegment(self):
        return PInfInterpolatedSegment(self.a,
                                   #ChebyshevInterpolator_PInf(self.f, self.a))
                                   PInfInterpolator(self.f, self.a))
    def findRightpoint(self):
        x = self.a+1
        fx = self.f(array([x]))
        h0 = fx
        while not ((fx <= h0 * params.segments.plot.yminEpsilon) or (abs(fx - self.f(array([x + 1.2 * abs(x-self.a)]))) <= h0 * params.segments.plot.yminEpsilon)):
            x = x + 1.2 * abs(x-self.a)
            fx = self.f(array([x]))
            if abs(x)>1e20:
                break
        return x
    def findRightEps(self):
        x = self.a + 0.1
        while (abs(self.f(x) - self.f(1.2*x))>1e-16):
            y = x+1.2*abs(x-self.a)
            if isnan(self.f(y)) | isinf(self.f(y)):
                break
            x = x+1.2*abs(x-self.a)
        return x
    def integrate(self, a = None, b = None):
        """definite integral over interval (c, d) \cub (a, b) """
        if a==None or a<self.a:
            a=self.a
        if b==None or isposinf(b):
            i,e = integrate_fejer2_pinf(self.f, a, exponent = params.integration_infinite.exponent)
        elif b>a:
            i,e = integrate_fejer2_pinf(self.f, a, b, exponent = params.integration_infinite.exponent)
        else:
            i,e = 0,0
        return i
    def cumint(self, y0 = 0.0):
        """indefinite integral over interval (a, x)"""
        return PInfSegment(self.a, partial(call_segint2, self, y0))
    def segIntegral(self, x):
        if isscalar(x):
            return self.integrate(a = x)
        else:
            return array([self.integrate(a = xi) for xi in x])
    def shiftAndScale(self, shift, scale):
        """It produce f((x - shift)/scale) for given f(x)
        """
        _1_scale = 1.0 / scale
        if scale > 0:
            return PInfSegment(self.a * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift))
        else:
            return MInfSegment(self.a * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift))
    def isPInf(self):
        return True
    def tailexp(self):
        return estimateTailExponent(self, pos = True)

class DiracSegment(Segment):
    """Segment = single Dirac function at point a
    f(x) = ya * delta(x-a), \int_{-inf}^{+inf} f(x) = ya
    """
    def __init__(self, a, ya):
        super(DiracSegment, self).__init__(a, a, ya)
    def __str__(self):
        return "{0}, ({1}, {2})".format(self.__class__.__name__, self.a, self.f)
    def __call__(self,x):
        if isscalar(x):
            y = (x==self.a) * 1
        else:
            y=zeros_like(x)
            y[x==self.a] = 1 # TODO it should be Inf or self.f
        return y
    def integrate(self, a, b):
        if (a<self.a and self.b<b) or (a==b==self.a):
            return self.f
        else:
            return 0
    def cumint(self, y0 = 0.0):
        """indefinite integral over interval (a, x)"""
        return Segment(self.a , self.b, partial(call_segint, self, y0))
    def plot(self,
             xmin = None,
             xmax = None,
             show_nodes = None,
             show_segments = None,
             numberOfPoints = None, **args):
        if xmin == None:
            xmin = self.a
        if xmax == None:
            xmax = self.b
        if xmin <= self.a <= xmax:
            plot([self.a, self.a],[0, self.f],'-', **args)
            plot([self.a],[self.f],'k^')
    def semilogx(self, **args):
        semilogx([self.a, self.a],[0, self.f],'-', **args)
        semilogx([self.a],[self.f],'k^')
    def toInterpolatedSegment(self):
        return self
    def shiftAndScale(self, shift, scale):
        """Produce f((x - shift)/scale) for given f(x)."""
        return DiracSegment(self.a * scale + shift, self.f)
    def isSegment(self):
        return False
    def isDirac(self):
        return True
    def diff(self):
        return self
    def roots(self):
        return array([])

class InterpolatedSegment(Segment):
    """Interpolated Segment on interval [a, b]
    """
    def __init__(self, a, b, interpolatorOfF):
        super(InterpolatedSegment, self).__init__(a, b, interpolatorOfF)
    def __str__(self):
        return "{0} ({3} pts.), [{1}, {2}]".format(self.__class__.__name__, self.a, self.b, len(self.f.getNodes()[0]))
    def plot(self,
             xmin = None,
             xmax = None,
             show_nodes = True,
             show_segments = True,
             numberOfPoints = params.segments.plot.numberOfPoints, **args):
        super(InterpolatedSegment, self).plot(xmin = xmin,
                   xmax = xmax,
                   show_nodes = show_nodes,
                   show_segments = show_segments,
                   numberOfPoints = numberOfPoints, **args)
        if show_nodes:
            if (xmin == None):
                xmin = self.findLeftpoint()
            else:
                xmin = max(xmin, self.a)
            if (xmax == None):
                xmax = self.findRightpoint()
            else:
                xmax = min(xmax, self.b)
            Xs, Ys = self.f.getNodes()
            Ys=Ys[Xs>=xmin]
            Xs=Xs[Xs>=xmin]
            Ys=Ys[Xs<=xmax]
            Xs=Xs[Xs<=xmax]
            plot(Xs, Ys, 'o', markersize = params.segments.plot.nodeMarkerSize)
    def semilogx(self, **args):
        numberOfPoints = params.segments.plot.numberOfPoints
        Xs, Ys = self.f.getNodes()
        xmin = self.a
        xmax = self.b
        if (xmin == -Inf):
            xmin = self.findLeftpoint()
        if (xmax == Inf):
            xmax = self.findRightpoint()
        Ys=Ys[Xs>=xmin]
        Xs=Xs[Xs>=xmin]
        Ys=Ys[Xs<=xmax]
        Xs=Xs[Xs<=xmax]
        xi = linspace(xmin, xmax, numberOfPoints)
        yi = self.f(xi)
        semilogx(xi, yi, '-', **args) #label='interp', linewidth=1)
    def diff(self):
        return InterpolatedSegment(self.a, self.b, self.f.diff())
    def roots(self):
        return self.f.roots()
    def trim(self, abstol=None):
        if abstol is None:
            abstol = params.interpolation.convergence.abstol
        return InterpolatedSegment(self.a, self.b, self.f.trim(abstol=abstol))

class SegmentWithPole(Segment):
    """Segment with pole on interval (a, b)
    """
    def __init__(self, a, b, f, left_pole = True):
        safe_a = safe_b = None
        if left_pole:
            if a == 0:
                safe_a = 0
            else:
                safe_a = a + abs(a) * finfo(float).eps
        if not left_pole:
            if b == 0:
                safe_b = 0
            else:
                safe_b = b - abs(b) * finfo(float).eps
        super(SegmentWithPole, self).__init__(a, b, f, safe_a, safe_b)
        self.left_pole = left_pole
    def integrate(self, a = None, b = None):
        """definite integral over interval (c, d) \cub (a, b) """
        if a == None or a < self.a :
            a = self.a
        if b == None or b > self.b:
            b = self.b
        if a == b:
            return 0.0
        if self.hasLeftPole() and a == self.a:
            i,e = integrate_fejer2_Xn_transformP(self, a, b)
        elif self.hasRightPole() and b == self.b:
            i,e = integrate_fejer2_Xn_transformN(self, a, b)
        elif self.hasLeftPole() and a != self.a:
            i1,e1 = integrate_fejer2_Xn_transformP(self, self.a, b)
            i2,e2 = integrate_fejer2_Xn_transformP(self, self.a, a)
            i = i1 - i2
            i,e = integrate_fejer2_Xn_transformP(self, a, b)
        elif self.hasRightPole() and b != self.b:
            i1,e1 = integrate_fejer2_Xn_transformN(self, a, self.b)
            i2,e2 = integrate_fejer2_Xn_transformN(self, b, self.b)
            i = i1 - i2
        else:
            i,e = integrate_fejer2(self, a, b)
        return i
    def shiftAndScale(self, shift, scale):
        """It produce f((x - shift)/scale) for given f(x)
        """
        _1_scale = 1.0 / scale
        if scale > 0:
            return SegmentWithPole(self.a * scale + shift, self.b * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift), left_pole = self.left_pole)
        else:
            return SegmentWithPole(self.b * scale + shift, self.a * scale + shift, partial(_shift_and_scale, self.f, _1_scale, shift), left_pole = not self.left_pole)
    def cumint(self, y0 = 0.0):
        """indefinite integral over interval (a, x)"""
        return SegmentWithPole(self.a , self.b, partial(call_segint, self, y0), left_pole = self.left_pole)
    def plot(self,
             xmin = None,
             xmax = None,
             show_nodes = True,
             show_segments = True,
             numberOfPoints = None, **args):
        leftRightEpsilon = params.segments.plot.leftRightEpsilon
        if numberOfPoints is None:
            numberOfPoints = params.segments.plot.numberOfPoints
        if (xmin == None):
            xmin = self.a
        if (xmax == None):
            xmax = self.b
        xmin = max(xmin, self.a)
        xmax = min(xmax, self.b)
        if xmin>=xmax:
            return
        if self.left_pole:
            xmin = xmin + leftRightEpsilon
            xi = xmin + logspace(log10(abs(leftRightEpsilon)), log10(abs(xmax-xmin)), numberOfPoints)
            if xi[-1] > self.b:
                xi[-1] = self.b
        if not self.left_pole:
            xmax = xmax - leftRightEpsilon
            xi = xmax - logspace(log10(abs(xmin-xmax)), log10(abs(leftRightEpsilon)), numberOfPoints)
            if xi[0] < self.a:
                xi[0] = self.a
        yi = self.f(xi)
        plot(xi, yi,'-', **args)
        # show segment
        if not self.isMorPInf() and show_segments:
            plot([xmin, xmin], [0, yi[0]], 'c--', linewidth=1)
            plot([xmax, xmax], [0, yi[-1]], 'c--', linewidth=1)
        # show vertical asympthote
        if self.left_pole:
            plot([xmin, xmin], [0, yi[0]], 'k-.', linewidth=1)
        else:
            plot([xmax, xmax], [0, yi[-1]], 'k-.', linewidth=1)

    def toInterpolatedSegment(self, NoR = None, NoL = None):
        return InterpolatedSegmentWithPole(self.a, self.b, self.f, self.left_pole)
        #InterpolatedSegment(self.a, self.b,
        #                           #ChebyshevInterpolator(self.f, self.a, self.b)) # przywraca stan oryginalny
        #                           #ChebyshevInterpolator1(self.f, self.a, self.b))
        #                           #ValTransformInterpolator(self.f, self.a, self.b))
        #                           LogTransformInterpolator(self.f, self.a, self.b))
    def hasLeftPole(self):
        return self.left_pole
    def hasRightPole(self):
        return not self.left_pole
    def hasPole(self):
        return True
    def diff(self):
        return self.toInterpolatedSegment().diff()

class InterpolatedSegmentWithPole(SegmentWithPole):
    """Segment with pole on interval (a, b) using log(f(exp(x))
    """
    def __str__(self):
        return "{0} ({3} pts.), [{1}, {2}]".format(self.__class__.__name__, self.a, self.b, len(self.f.getNodes()[0]))
    def __init__(self, a, b, f, left_pole=True, exponent = 1.0, residue = 0.0):
        if left_pole:
            f = PoleInterpolatorP(f, a, b, par = params.interpolation_pole)
        else:
            f = PoleInterpolatorN(f, a, b, par = params.interpolation_pole)
        SegmentWithPole.__init__(self, a, b, f, left_pole)
    def hasPole(self):
        return True

    def plot(self,
             xmin = None,
             xmax = None,
             show_nodes = True,
             show_segments = True,
             numberOfPoints = None, **args):
        if numberOfPoints is None:
            numberOfPoints = params.segments.plot.numberOfPoints
        super(InterpolatedSegmentWithPole, self).plot(xmin = xmin,
                   xmax = xmax,
                   show_nodes = show_nodes,
                   show_segments = show_segments,
                   numberOfPoints = numberOfPoints, **args)
        if show_nodes:
            leftRightEpsilon = params.segments.plot.leftRightEpsilon
            if (xmin == None):
                xmin = self.findLeftpoint()
            else:
                xmin = max(xmin, self.a)
            if (xmax == None):
                xmax = self.findRightpoint()
            else:
                xmax = min(xmax, self.b)
            Xs, Ys = self.f.getNodes()
            if self.hasLeftPole():
                xmin = xmin + leftRightEpsilon
            if self.hasRightPole():
                xmax = xmax - leftRightEpsilon
            Ys=Ys[Xs>=xmin]
            Xs=Xs[Xs>=xmin]
            Ys=Ys[Xs<=xmax]
            Xs=Xs[Xs<=xmax]
            plot(Xs, Ys, 'o', markersize = params.segments.plot.nodeMarkerSize)
    def diff(self):
        return InterpolatedSegment(self.a, self.b, self.f.diff())
# TODO: to remove ???
class InterpolatedSegmentWithPole2(InterpolatedSegmentWithPole):
    """Segment with pole on interval (a, b) using interpolation f(x)/(x-pole)^exponent
    """
    def __init__(self, a, b, f, left_pole = True, exponent = 1.0, residue = 0.0):
        self.exponent = exponent
        #self.f = ChebyshevInterpolator(lambda x : nan_to_num(lambda x : f(x) * abs(x-self.pole) ** exponent, x, residue), a, b) # przywraca stan oryginalny
        self.f = ChebyshevInterpolator1(lambda x : nan_to_num(lambda x : f(x) * abs(x-self.pole) ** exponent, x, residue), a, b)
        #self.f = ValTransformInterpolator(lambda x : nan_to_num(lambda x : f(x) * abs(x-self.pole) ** exponent, x, residue), a, b)
        super(InterpolatedSegmentWithPole2, self).__init__(a, b, self.f, left_pole) # , xp, exponent

    def __call__(self, x):
        if isscalar(x):
            x=array(x)
        ind = where((x>=self.a) & (x<=self.b))
        y = zeros_like(x)
        y[ind] = self.f(x[ind])/abs(x[ind]-self.pole) ** self.exponent
        return y

class MInfInterpolatedSegment(InterpolatedSegment, MInfSegment):
    """Interpolated Segment on interval [-inf, b]
    """
    def __init__(self, b, interpolatorOfF):
        MInfSegment.__init__(self, b, interpolatorOfF)
    def diff(self):
        return MInfInterpolatedSegment(self.a, self.f.diff())
class PInfInterpolatedSegment(InterpolatedSegment, PInfSegment):
    """Interpolated Segment on interval [a, inf]
    """
    def __init__(self, a, interpolatorOfF):
        PInfSegment.__init__(self, a, interpolatorOfF)
    def diff(self):
        return PInfInterpolatedSegment(self.a, self.f.diff())
class breakPoint(object):
    """Decribe a breakpoint of a piecewise function."""
    def __init__(self, x, negPole, posPole, Cont = True, dirac = False):
        self.x = x
        self.negPole = negPole
        self.posPole = posPole
        self.Cont = Cont
        self.dirac = dirac
    def __str__(self):
        ret = ["[" + str(self.x)]
        if self.negPole:
            ret.append("negPole")
        if self.posPole:
            ret.append("posPole")
        if not self.Cont and not self.negPole and not self.posPole:
            ret.append("Discontinuous")
        if self.dirac:
            ret.append("Dirac")
        return ",".join(ret) + "]"

class PiecewiseFunction(object):
    """Base object for piecewise functions.
    """
    def __init__(self, segments = [], fun = None, breakPoints = [], lpoles = None, rpoles = None):
        """It make blank PiecewiseFunction object, if fun is None
        or PiecewiseFunction for a given fun, break points otherwise.

        """
        if lpoles is None:
            self.lpoles = zeros_like(breakPoints)>0.0
        else:
            self.lpoles = lpoles
        if rpoles is None:
            self.rpoles = zeros_like(breakPoints)>0.0
        else:
            self.rpoles = rpoles
        self.breaks=[]
        if fun is not None:
            assert(len(breakPoints)>1 and breakPoints[0]<breakPoints[1]<=breakPoints[-1]), breakPoints
            assert(len(breakPoints) == len(self.rpoles) == len(self.lpoles))
            self.breaks = breakPoints
            self.segments = []
            if (isinf(self.breaks[0])):
                self.addSegment(MInfSegment(self.breaks[1],fun))
            else:
                if self.lpoles[0] == False and self.rpoles[1] == False:
                    self.addSegment(Segment(self.breaks[0], self.breaks[1], fun))
                elif self.lpoles[0] == True and self.rpoles[1] == False:
                    self.addSegment(SegmentWithPole(self.breaks[0], self.breaks[1], fun, left_pole = True))
                elif self.rpoles[0] == False and self.rpoles[1] == True:
                    self.addSegment(SegmentWithPole(self.breaks[0], self.breaks[1], fun, left_pole = False))
                else:
                    assert(False) #  no segment of such type
            if len(breakPoints)==2:
                return
            for i in range(1, len(breakPoints)-2):
                #self.addSegment(Segment(self.breaks[i],self.breaks[i+1], fun))
                if self.lpoles[i] == False and self.rpoles[i+1] == False:
                    self.addSegment(Segment(self.breaks[i], self.breaks[i+1], fun))
                elif self.lpoles[i] == True and self.rpoles[i+1] == False:
                    self.addSegment(SegmentWithPole(self.breaks[i], self.breaks[i+1], fun, left_pole = True))
                elif self.rpoles[i] == False and self.rpoles[i+1] == True:
                    self.addSegment(SegmentWithPole(self.breaks[i], self.breaks[i+1], fun, left_pole = False))
                else:
                    assert(False) # no segment of such type
            if (isinf(self.breaks[-1])):
                self.addSegment(PInfSegment(self.breaks[-2], fun))
            else:
                #self.addSegment(Segment(self.breaks[-2], self.breaks[-1], fun))
                if self.lpoles[-2] == False and self.rpoles[-1] == False:
                    self.addSegment(Segment(self.breaks[-2], self.breaks[-1], fun))
                elif self.lpoles[-2] == True and self.rpoles[-1] == False:
                    self.addSegment(SegmentWithPole(self.breaks[-2], self.breaks[-1], fun, left_pole = True))
                elif self.rpoles[-2] == False and self.rpoles[-1] == True:
                    self.addSegment(SegmentWithPole(self.breaks[-2], self.breaks[-1], fun, left_pole = False))
                else:
                    assert(False) # no segment of such type
        else:
            # TODO to remove
            self.segments = segments
            if len(segments)>0 :
                i=0
                for seg in self.segments:
                    self.breaks[i]=seg.a
                    i=i+1;
                self.breaks[i]=seg.b

    def __call__(self, x):
        if isscalar(x):
            # binary search
            segment = self.findSegment(x)
            if segment is None :
                return 0
            else:
                if segment.isDirac():
                    return segment(x)
                else:
                    return segment.f(x)
            return None
        else:
            # iterate over segments
            y = zeros_like(x)
            for seg in self.getSegments():#segments:
                ind = ((x>=seg.a) & (x<seg.b))
                if ind.any():
                    y[ind] = seg.f(x[ind])
            if len(self.getSegments())>0:
                ind = (x==self.getSegments()[-1].b)
                y[ind] = seg.f(x[ind])
            return y

    def addSegment(self, segment):
        """It insert segment in proper order"""
        if segment.isDirac():
            for seg in self.getDiracs():
                if abs(segment.a - seg.a) <= params.segments.unique_eps:
                    seg.f += segment.f
                    return
        if len(self.segments) > bisect.bisect_left(self.segments, segment):
            seg = self.segments[bisect.bisect_left(self.segments, segment)]
            if not (seg>segment or segment>seg):
                if segment.isDirac():
                    segment = DiracSegment(seg.a, segment.f)
                assert seg>segment or segment>seg, "{}=={}=={}=={}=={}".format(self.segments, seg, segment, seg.a, segment.a)
        bisect.insort(self.segments, segment)
        self.breaks = unique(append(self.breaks, [segment.a, segment.b]))
    def findSegment(self, x):
        """It return segment containing point x"""
        xsegment = DiracSegment(x,0)
        ind = bisect.bisect_left(self.segments, xsegment)
        if  0 < ind < len(self.segments):
            return self.segments[ind]
        elif ind == 0:
            if self.segments[ind].a <= x <= self.segments[ind].b :
                return  self.segments[ind]
            else :
                return None
        elif ind == len(self.segments):
            if self.segments[ind-1].b == x:
                return  self.segments[ind-1]
            else :
                return None
        else:
            assert(False)

    def findSegmentByEnds(self, a, b):
        """Return the segment with end points (a,b)"""
        for seg in self.segments:
            if seg.a == a and seg.b == b:
                return seg
        return None

    def toInterpolated(self):
        interpolatedPFun = self.__class__([])
        for seg in self.segments:
            interpolatedPFun.addSegment(seg.toInterpolatedSegment())
        return interpolatedPFun

    def integrate(self, a = None, b = None):
        I = 0.0
        if a==None:
            a = -Inf
        if b==None:
            b = +Inf
        for seg in self.segments:
            i = seg.integrate(a, b)
            I = I + i
        return I
    def getInterpErrors(self):
        err = list()
        for seg in self.getSegments():
            if isinstance(seg.f, AdaptiveInterpolator) or isinstance(seg.f, AdaptiveInterpolator1):
                err.append((seg.a, seg.b, seg.f.err))
            elif isinstance(seg.f, PInfInterpolator):
                err.append((seg.a, seg.b, seg.f.vb.err))
            else:
                err.append((seg.a, seg.b, 0))
        return err
    def isNonneg(self):
        for seg in self.segments:
            if seg.isSegment():
                if seg.a < 0:
                    return False
            if seg.isDirac():
                if seg.a <= 0:
                    return False
        return True

    def cumint(self):
        integralPFun = CumulativePiecewiseFunction([])
        f0 = 0
        for seg in self.segments:
            if seg.isPInf():
                f0 = f0 + seg.integrate(seg.a)
            segi = seg.cumint(f0)
            if seg.isDirac():
                f0 = f0 + seg.f
            else:
                integralPFun.addSegment(segi)
            if not seg.isPInf():
                f0 = segi.f(segi.b)
        rightval = max(f0, segi.f(segi.b))
        if not segi.isPInf():
            integralPFun.addSegment(PInfSegment(segi.b, ConstFun(rightval)))
        if not integralPFun.segments[0].isMInf():
            integralPFun.addSegment(MInfSegment(integralPFun.segments[0].a, ConstFun(0)))
        return integralPFun
    def diff(self):
        diffPFun = PiecewiseFunction([])
        for seg in self.segments:
            if seg.isMInf() or seg.isPInf():
                pass
                #raise NotImplementedError("not implemented")
            else:
                segi = seg.diff()
                diffPFun.addSegment(segi)
        return diffPFun
    def roots(self):
        r = array([])
        for seg in self.segments:
            if not seg.isDirac():
                r = concatenate((r, seg.roots()))
        return r
    def characteristicPoints(self):
        mi, xi = array([]), array([])
        df = self.diff()
        xi = df.roots()
        mi = self(xi);
        segVals = self.getSegVals()
        xi = concatenate((xi, self.getBreaks()[0:-1], self.getBreaks()[1:]))
        mi = concatenate((mi, [v[0]  for v in segVals], [v[1]  for v in segVals] ))
        return mi, xi
    def max_(self):
        mi, xi = self.characteristicPoints()
        return max(mi), xi[argmax(mi)]
    def min_(self):
        mi, xi = self.characteristicPoints()
        return min(mi), xi[argmin(mi)]
    def trimInterpolators(self, abstol=None):
        diffPFun = PiecewiseFunction([])
        if abstol is None:
            abstol = params.interpolation.convergence.abstol
        for seg in self.segments:
            segi = seg.trim(abstol=abstol)
            diffPFun.addSegment(segi)
        return diffPFun

    def ccumint(self):
        """TODO complementary cumint i.e int_x^Inf f(x) dx"""
        integralPFun = CumulativePiecewiseFunction([])
        f0 = 0
        for seg in self.segments:
            if seg.isPInf():
                f0 = f0 + seg.integrate(seg.a)
            segi = seg.cumint(f0)
            if seg.isDirac():
                f0 = f0 + seg.f
            else:
                integralPFun.addSegment(segi)
            if not seg.isPInf():
                f0 = segi.f(segi.b)
        rightval = max(f0, segi.f(segi.b))
        if not segi.isPInf():
            integralPFun.addSegment(PInfSegment(segi.b, ConstFun(rightval)))
        if not integralPFun.segments[0].isMInf():
            integralPFun.addSegment(MInfSegment(integralPFun.segments[0].a, ConstFun(0)))
        return integralPFun
    def max_abs(self):
        x1, m1 = self.maximum()
        x2, m2 = self.minimum()
        if abs(m1)<abs(m2):
            return x2, abs(m2)
        else:
            return x1, m1
    def maximum(self):
        """Mode, using scipy's function fminbound, may be inaccurate """
        if not have_Scipy_optimize:
            print("Warning: scipy's fminbound not found")
            return None
        m = -inf
        x = None
        for seg in self.segments:
            if not seg.isDirac() :
                if seg.hasLeftPole():
                    xi = fminbound(lambda x: -seg(x) + 1e-14, seg.a, seg.b, xtol = 1e-16)
                elif seg.hasRightPole() :
                    xi = fminbound(lambda x: -seg(x), seg.a + 1e-14, seg.b, xtol = 1e-16)
                else:
                    xi = fminbound(lambda x: -seg(x), seg.a, seg.b, xtol = 1e-16)
                mi = float(seg.f(xi))
            else:
                xi, mi = seg.a, seg.f
            if m < mi:
                m = mi
                x = xi
        return x, m
    def minimum(self):
        """Mode, using scipy's function fminbound, may be inaccurate """
        if not have_Scipy_optimize:
            print("Warning: scipy's fminbound not found")
            return None
        m = inf
        x = None
        for seg in self.segments:
            if not seg.isDirac() :
                if seg.hasLeftPole():
                    xi = fminbound(seg, seg.a, seg.b, xtol = 1e-16)
                elif seg.hasRightPole() :
                    xi = fminbound(seg, seg.a + 1e-14, seg.b, xtol = 1e-16)
                else:
                    xi = fminbound(seg, seg.a, seg.b, xtol = 1e-16)
                mi = float(seg.f(xi))
            else:
                xi, mi = seg.a, seg.f
            if m > mi:
                m = mi
                x = xi
        return x, m
    def __str__(self):
        return ','.join(['({0})'.format(str(seg)) for seg in self.segments])

    def plot(self,
             xmin = None,
             xmax = None,
             cl = None,
             show_nodes = None,
             show_segments = None,
             numberOfPoints = None, **args):
        if cl is None:
            cl = params.segments.plot.ciLevel
        if show_nodes is None:
            show_nodes = params.segments.plot.showNodes
        if show_segments is None:
            show_segments = params.segments.plot.showSegments
        if numberOfPoints is None:
            numberOfPoints = params.segments.plot.numberOfPoints
        if not params.segments.plot.showSegments and "color" not in args:
            args["color"] = "k"
        if cl is not None and xmin is None and xmax is None:
            xmin, xmax = self.ci(cl)
        h0 = h1 = 0
        for i, seg in enumerate(self.segments):
            xi = seg.a
            try:
                if seg.isDirac():
                    h1 = seg.f
                else:
                    if xi+1e-10 >= seg.b:
                        h1 = float(seg.f((seg.a + seg.b)/2))
                    else:
                        h1 = float(seg.f(xi+1e-10))
            except Exception as e:
                h1 = 0.0
                h0 = 0.0
            seg.plot(xmin = xmin,
                     xmax = xmax,
                     show_nodes = show_nodes,
                     show_segments = show_segments,
                     numberOfPoints = numberOfPoints, **args)
            if (not seg.isMInf()) and (not seg.hasLeftPole()) and (xmin is None or xmin<=xi) and (xmax is None or xi<=xmax):
                plot([xi,xi], [h0, h1], 'k--')
            try:
                if seg.b-1e-10 <= seg.a:
                    h0 = float(seg.f((seg.a + seg.b)/2))
                else:
                    h0 = float(seg.f(seg.b-1e-10))
            except Exception as e:
                h0 = 0.0
            # plot right dashed line if next segment further off
            if i < len(self.segments) - 1 and self.segments[i+1].a > seg.b + params.segments.unique_eps:
                if (not seg.isPInf()) and (xmin is None or xmin<=xi) and (xmax is None or xi<=xmax):
                    plot([seg.b,seg.b], [h0, 0], 'k--')
                    h0 = 0.0
            if "label" in args:
                # avoid a label in legend for every segment
                del args["label"]
        seg = self.segments[-1]
        xi = seg.b
        if (not seg.isPInf()) and (xmin is None or xmin<=xi) and (xmax is None or xi<=xmax):
            plot([xi,xi], [h0, 0], 'k--')
    def getPiecewiseSpace(self,
             xmin = None,
             xmax = None,
             numberOfPoints = None, **args):
        if numberOfPoints is None:
            numberOfPoints = params.segments.plot.numberOfPoints
        xi =array([])
        for seg in self.segments:
            xi = concatenate([xi,seg.getSegmentSpace(xmin = xmin,
                   xmax = xmax,
                   numberOfPoints = numberOfPoints, **args)])
        return xi
    def semilogx(self, **args):
        for x in self.segments:
            x.semilogx(**args)

    def plot_tails(self, maxxexp = 20, asympf = None):
        subplot(122)
        X = logspace(-1,maxxexp, 10000)
        Y = self(X)
        if not (Y==0).all():
            if asympf is None:
                loglog(X, Y)
            Xs, Ys = self.segments[-1].f.getNodes()
            Xs = Xs[1:-1]
            Ys = Ys[1:-1]
            if asympf is not None:
                Ys = log(Ys)
                Ys -= asympf(log(Xs))
                semilogx(Xs, Ys, "o")
            else:
                loglog(Xs, Ys, "o")
        subplot(121)
        Y = self(-X)
        if not (Y==0).all():
            if asympf is None:
                loglog(X, Y)
            Xs, Ys = self.segments[0].f.getNodes()
            Xs = Xs[1:]
            Ys = Ys[1:]
            if asympf is not None:
                Ys = log(Ys)
                Ys -= asympf(log(-Xs))
                semilogx(-Xs, Ys, "o")
            else:
                loglog(-Xs, Ys, "o")

    def copyShiftedAndScaled(self, shift = 0, scale = 1):
        copyFunction = self.__class__([])
        for seg in self.segments:
            copyFunction.addSegment(seg.shiftAndScale(shift, scale))
        return copyFunction
    def copyComposition(self, f, finv, finvDeriv, pole_at_zero = False):
        """Composition with injective function f"""
        self_with_zero = self.splitByPoints([0])
        copyFunction = self.__class__([])
        for seg in self_with_zero.segments:
            segcomp = seg.probComposition(f, finv, finvDeriv, pole_at_zero = pole_at_zero)
            copyFunction.addSegment(segcomp)
        return copyFunction

    def copyCompositionNoninjective(self, intervals, fs, finvs, finvDerivs, pole_at_zero = False):
        """Composition with non-injective function f """
        assert len(fs)==len(finvs)==len(finvDerivs), "f, finv, finvDeriv should be lists of the same size"
        #self_with_zero = self.splitByPoints([0])
        copyFunction = self.__class__([])
        a, b = self.range()
#        intervals_ = copy(intervals)
#        if intervals[0][0]<=a and b<=intervals[-1][0]:
#            # ok
#        else:
#            if a<=intervals[0][0]:
#
#            pass
#        if a<=intervals[0][0]:
#            pass

        for i in range(len(fs)):
            summand = self.restrictToInterval(intervals[i][0], intervals[i][1])
            copyFunction = copyFunction + summand.copyComposition(fs[i], finvs[i], finvDerivs[i], pole_at_zero)
        return copyFunction

    def symerticalNested(self):
        # TODO
        pass

    def copyProbInverse(self, pole_at_zero = False):
        """Composition with 1/x"""
        self_with_zero = self.splitByPoints([0])
        copyFunction = self.__class__([])
        for seg in self_with_zero.segments:
            segcomp = seg.probInverse(pole_at_zero = pole_at_zero)
            copyFunction.addSegment(segcomp)
        return copyFunction
    def copySquareComposition(self):
        """Composition with x^2"""
        fun = self.splitByPoints([0.0])
        copyFunction = self.__class__([])
        leftFunction = self.__class__([])
        rightFunction = self.__class__([])
        for seg in fun.segments:
            if seg.a >= 0:
                leftFunction.addSegment(seg.squareComposition())
            else:
                rightFunction.addSegment(seg.squareComposition())
        if len(rightFunction.segments) == 0:
            copyFunction = leftFunction
        elif len(leftFunction.segments) == 0:
            copyFunction = rightFunction
        else:
            copyFunction = leftFunction + rightFunction
        return copyFunction
    def copyAbsComposition(self):
        """Composition with |x|"""
        fun  =self.splitByPoints([0.0])
        leftFunction = self.__class__([])
        rightFunction = self.__class__([])
        for seg in fun.segments:
            if seg.a >= 0:
                leftFunction.addSegment(seg.absComposition())
            else:
                rightFunction.addSegment(seg.absComposition())
        leftFunction = leftFunction.splitByPoints(rightFunction.getBreaks())
        #leftFunction.add_diracs(rightFunction)
        copyFunction = leftFunction + rightFunction
        return copyFunction
    def copyLogComposition(self, f, finv, finvDeriv, pole_at_zero = False):
        """Composition with logarithm"""
        fun  =self.splitByPoints([0.5, 1.0, 2.0])
        return fun.copyComposition(f, finv, finvDeriv, pole_at_zero = False)

    def getBreaksExtended(self):
        # a version which reports poles on the left/right of each breakpoint, etc.
        #segments = self.getSegments()
        segments = self.segments
        if len(segments) == 0:
            return []
        breaks = [breakPoint(segments[0].a, False, False)]
        for seg in segments:
            if seg.isDirac():
                breaks[-1].dirac = True
            if seg.hasLeftPole():
                breaks[-1].posPole = True
            if seg.hasRightPole():
                breaks.append(breakPoint(seg.b, True, False))
            else:
                breaks.append(breakPoint(seg.b, False, False))
        # check continuity
        seg = segments[0]
        if not seg.isDirac() and not isinf(seg.a) and (seg.hasLeftPole() or seg.f(seg.a) > params.pole_detection.continuity_eps):
            breaks[0].Cont = False
        seg = segments[-1]
        if not seg.isDirac() and not isinf(seg.b) and (seg.hasRightPole() or seg.f(seg.b) > params.pole_detection.continuity_eps):
            breaks[-1].Cont = False
        for i, seg in enumerate(segments[1:]):
            if not seg.isDirac() and not segments[i].isDirac():
                if not segments[i].hasRightPole() and not seg.hasLeftPole():
                    if (segments[i].hasRightPole() or seg.hasLeftPole()
                        or abs(segments[i].f(self.segments[i].b) - seg.f(seg.a)) > params.pole_detection.continuity_eps):
                        breaks[i+1].Cont = False
        return breaks

    def getBreaks(self):
        if len(self.segments) == 0:
            return zeros(len(self.segments))
        else:
            breaks = unique([seg.a for seg in self.segments] + [seg.b for seg in self.segments])
            return breaks

    def getDiracs(self, xi = None):
        diracs = []
        for seg in self.segments:
            if seg.isDirac():
                diracs.append(seg)
        return diracs

    def getSegVals(self):
        val_list = []
        for seg in self.segments:
            left = seg(seg.a)
            right = seg(seg.b)
            val_list = val_list + [(left, right)]
#        for seg in self.segments:
#            if seg.isMInf():
#                val_list = val_list + [(seg.f(seg.findLeftEps()), seg.f(seg.b))]
#            elif seg.isPInf():
#                val_list = val_list + [(seg.f(seg.a+1e-14), seg.f(seg.findRightEps()))]
#            else:
#                if not isscalar(seg.f):
#                    left = seg.f(seg.a)
#                else:
#                    left = seg.f
#                if not isscalar(seg.f):
#                    right= seg.f(seg.b)
#                else:
#                    right= seg.f
#                if not isfinite(left):
#                    if len(val_list)>0:
#                        left  = val_list[-1][1]
#                val_list = val_list + [(left, right)]
        return val_list

    def getDirac(self, xi):
        for seg in self.segments:
            if seg.isDirac() and seg.a == xi:
                return seg
        return None
    def getSegments(self):
        segments = []
        for seg in self.segments:
            if seg.isSegment():
                segments.append(seg)
        return segments
    def printtex(self):
        str = "\\begin{tabular}{|r|l|c|c|c|}\hline\n"
        i = 0
        row = '$i$ & type & $a_i$ & $b_i$ & $n_i$ \\\\ \\hline\\hline\n'
        str+=row
        for seg in self.segments:
            i +=1
            try:
                ni = len(seg.f.Xs)
            except :
                ni = "-"
            row = '{0} & {1} & {2} & {3}  & {4}  \\\\ \\hline \n'.format(i, seg.__class__.__name__, seg.a, seg.b, ni)
            str+=row
        str += "\\end{tabular}"
        print(str)

    def add_diracs(self, other):
        """Pointwise sum of discrete part of two piecewise functions """
        for other_dirac in other.getDiracs():
            self_dirac = self.getDirac(other_dirac.a)
            if self_dirac is None:
                self.addSegment(other_dirac)
            else:
                self_dirac.f += other_dirac.f


    def __add__(self, other):
        if isinstance(other, PiecewiseFunction):
            return self._operation__(other, operation = operator.add)
        elif isinstance(other, numbers.Real):
            return self.__radd__(other)
        return NotImplemented
    def __sub__(self, other):
        if isinstance(other, PiecewiseFunction):
            return self._operation__(other, operation = operator.sub)
        elif isinstance(other, numbers.Real):
            return self.__rsub__(other)
        return NotImplemented
    def __mul__(self, other):
        if isinstance(other, PiecewiseFunction):
            return self._operation__(other, operation = operator.mul)
        elif isinstance(other, numbers.Real):
            return self.__rmul__(other)
        return NotImplemented
    def __truediv__(self, other):
        #TODO handle zeros in denominator, !!!currently unhandled!!!
        if isinstance(other, PiecewiseFunction):
            return self._operation__(other, operation = operator.div)
        elif isinstance(other, numbers.Real):
            return self.__rtruediv__(other)
        return NotImplemented
    def __div__(self, other):
        #TODO handle zeros in denominator, !!!currently unhandled!!!
        if isinstance(other, PiecewiseFunction):
            return self._operation__(other, operation = operator.div)
        elif isinstance(other, numbers.Real):
            return self.__rdiv__(other)
        return NotImplemented
    def __radd__(self, r, operation = operator.add):
        """Overload sum with real number: distribution of r+X."""
        if isinstance(r, numbers.Real):
            return self._roperation__(r, operation)
        return NotImplemented
    def __rsub__(self, r, operation = _op_rsub):
        """Overload sum with real number: distribution of r-X."""
        if isinstance(r, numbers.Real):
            return self._roperation__(r, operation)
        return NotImplemented
    def __rmul__(self, r, operation = operator.mul):
        """Overload product with real number: distribution of r*X."""
        if isinstance(r, numbers.Real):
            return self._roperation__(r, operation = operator.mul)
        return NotImplemented
    def __rtruediv__(self, r, operation = _op_rdiv):
        """Overload division by real number: distribution of r/X."""
        if isinstance(r, numbers.Real):
            return self._roperation__(r, operation )
        return NotImplemented
    def __rdiv__(self, r, operation = _op_rdiv):
        """Overload division by real number: distribution of r/X."""
        if isinstance(r, numbers.Real):
            return self._roperation__(r, operation )
        return NotImplemented

    def _roperation__(self, r, operation = operator.add):
        """Pointwise sum piecewise functions with real number """
        breaks = self.getBreaks()
        segsList = self.segments
        fun = self.__class__([])
        for seg in segsList:
            seg_fun = partial(_segfun_scalar_op, seg, operation, r)
            if seg.isMInf():
                fun.addSegment(MInfSegment(seg.b, seg_fun))
            elif seg.isPInf():
                fun.addSegment(PInfSegment(seg.a, seg_fun))
            elif seg.isSegment() and seg.hasLeftPole():
                fun.addSegment(SegmentWithPole(seg.a, seg.b, seg_fun, left_pole = True))
            elif seg.isSegment() and seg.hasRightPole():
                fun.addSegment(SegmentWithPole(seg.a, seg.b, seg_fun, left_pole = False))
            elif seg.isSegment():
                si=Segment(seg.a, seg.b, seg_fun)
                fun.addSegment(si)
            elif seg.isDirac():
                si = DiracSegment(seg.a, operation(seg.f, r))
                fun.addSegment(si)
            else:
                assert False
        return fun

    def op_diracs(self, other, op):
        """Pointwise operation on discrete parts of two piecewise functions."""
        ai = []
        for tmp in self.getDiracs():
            ai.append(tmp.a)
        for tmp in other.getDiracs():
            if not tmp.a in set(ai):
                ai.append(tmp.a)
        for dirac in ai:#other.getDiracs():
            self_dirac = self.getDirac(dirac)
            other_dirac = other.getDirac(dirac)
            if self_dirac is None:
                self.addSegment(DiracSegment(other_dirac.a, other_dirac.f))
            elif other_dirac is not None:
                self_dirac.f = op(self_dirac.f,  other_dirac.f)

    def _operation__(self, other, operation = operator.add):
        """Pointwise operation on two piecewise functions."""
        breaks1 = self.getBreaks()
        breaks2 = other.getBreaks()
        breaks = unique(concatenate((breaks1, breaks2)))
        f = self.splitByPoints(breaks)
        g = other.splitByPoints(breaks)
        segsList = _makeCommonSegmentList(f, g)

        fun = self.__class__([])
        def make_segfun(segment1, segment2):
            if segment1 is None:
                return partial(_scalar_segfun_op, segment2, operation, 0)
            elif segment2 is None:
                return partial(_segfun_scalar_op, segment1, operation, 0)
            else:
                return partial(_segfun_segfun_op, segment1, segment2, operation)
        for segf, segg in segsList:
            if segf is None:
                seg = segg
            elif segg is None:
                seg = segf
            else:
                seg = segf
            if seg is None: # only diracs
                continue
            if seg.isMInf():
                fun.addSegment(MInfSegment(seg.b, make_segfun(segf, segg)))
            elif seg.isPInf():
                fun.addSegment(PInfSegment(seg.a, make_segfun(segf, segg)))
            elif seg.isSegment() and seg.hasLeftPole():
                fun.addSegment(SegmentWithPole(seg.a, seg.b, make_segfun(segf, segg), left_pole = True))
            elif seg.isSegment() and seg.hasRightPole():
                fun.addSegment(SegmentWithPole(seg.a, seg.b, make_segfun(segf, segg), left_pole = False))
            elif seg.isSegment():
                si=Segment(seg.a, seg.b, make_segfun(segf, segg))
                fun.addSegment(si)
            else:
                assert False
        fun.op_diracs(self, operation)
        fun.op_diracs(other, operation)
        return fun

    def __pow__(self, k):
        return self._function__(fun = lambda x : x**k)
    def _log(self):
        #TODO handle zeros in arguments
        return self._function__(fun = numpy.log)
    def exp(self):
        return self._function__(fun = numpy.exp)
    def _log1(self):
        #TODO handle zeros in arguments
        return self._function__(fun = lambda x : -numpy.log(x))

    def _function__(self, fun = lambda x: x):
        """Pointwise sum of two piecewise functions."""
        result = self.__class__([])
        for seg in self.segments:
            safe_fun = partial(_safe_fun_on_f_call, seg, fun)
            if seg.isMInf():
                result.addSegment(MInfSegment(seg.b, safe_fun))
            elif seg.isPInf():
                result.addSegment(PInfSegment(seg.a, safe_fun))
            elif seg.isSegment() and seg.hasLeftPole():
                result.addSegment(SegmentWithPole(seg.a, seg.b, safe_fun, left_pole = True))
            elif seg.isSegment() and seg.hasRightPole():
                result.addSegment(SegmentWithPole(seg.a, seg.b, safe_fun, left_pole = False))
            elif seg.isSegment():
                si=Segment(seg.a, seg.b, safe_fun)
                result.addSegment(si)
            elif seg.isDirac():
                si=DiracSegment(seg.a, fun(seg.f))
                result.addSegment(si)
            else:
                assert False
        return result
    def restrictToInterval(self, a, b):
        """return function restricted to interval [a,b)
        """
        source = self.splitByPoints([a,b])
        fun = self.__class__([])
        for seg in source.segments:
            if seg.isDirac():
                if a<=seg.a and seg.b<b:
                    fun.addSegment(seg)
            else:
                if a<=seg.a and seg.b<=b:
                    fun.addSegment(seg)
        return fun
    def splitByPoints(self, points):
        """Pointwise subtraction of two piecewise functions """
        points = array(points)
        fun = self.__class__([])
        for seg in self.segments:
            inds = points[(seg.a<points) & (points<seg.b)]
            a = seg.a
            b = None
            wrapped_f = wrap_f(seg.f)
            for ind in inds:
                b = ind
                if seg.isMInf() and isinf(a):
                    fun.addSegment(MInfSegment(b, wrapped_f))
                elif seg.isPInf() and isinf(b):
                    fun.addSegment(PInfSegment(a, wrapped_f))
                elif seg.hasLeftPole() and a == seg.a:
                    fun.addSegment(SegmentWithPole(a, b, wrapped_f, left_pole = True))
                elif seg.hasRightPole() and b == seg.b:
                    fun.addSegment(SegmentWithPole(a, b, wrapped_f, left_pole = False))
                elif seg.isDirac():
                    fun.addSegment(DiracSegment(a, wrapped_f))
                else:
                    fun.addSegment(Segment(a, b, wrapped_f))
                a = b
            b = seg.b
            if seg.isMInf() and isinf(a):
                fun.addSegment(MInfSegment(b, wrapped_f))
            elif seg.isPInf() and isinf(b):
                fun.addSegment(PInfSegment(a, wrapped_f))
            elif seg.hasLeftPole() and a == seg.a:
                fun.addSegment(SegmentWithPole(a, b, wrapped_f, left_pole = True))
            elif seg.hasRightPole() and b == seg.b:
                fun.addSegment(SegmentWithPole(a, b, wrapped_f, left_pole = False))
            elif seg.isDirac():
                fun.addSegment(DiracSegment(a, wrapped_f))
            else:
                fun.addSegment(Segment(a, b, wrapped_f))
        return fun
    def inverse_scalar(self, y, fix_for_cdf=True):
        vals = self.getSegVals()
        breaks = self.getBreaks()
        x = None
        # continuous part of cumilative function
        for i in range(len(vals)):
            segi = self.segments[i]
            #if (vals[i][0]<=y<=vals[i][1]):
            if (vals[i][0]<=y<=vals[i][1] or vals[i][0]>=y>=vals[i][1]):
                if segi.isMInf():
                    #x = findinv(segi.f, a = segi.findLeftEps(), b = segi.b, c = y, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter)
                    if y == 0:
                        x = -inf
                    else:
                        x = findinv_minf(segi.f, b = segi.b, c = y, rtol = params.segments.cumint.reltol, xtol=params.segments.cumint.abstol, maxiter = params.segments.cumint.maxiter)
                elif segi.isPInf():
                    #x = findinv(segi.f, a = segi.a, b = segi.findRightEps(), c = y, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter)
                    if y == 1:
                        x = inf
                    else:
                        x = findinv_pinf(segi.f, a = segi.a, c = y, rtol = params.segments.cumint.reltol, xtol=params.segments.cumint.abstol, maxiter = params.segments.cumint.maxiter)
                else:
                    x = findinv(segi.f,  a = segi.a, b = segi.b, c = y, rtol = params.segments.cumint.reltol, xtol=params.segments.cumint.abstol, maxiter = params.segments.cumint.maxiter) # TODO PInd, MInf
        # discrete part of cumilative function
        for i in range(len(vals)-1):
            if (vals[i][1]<=y) & (y<=vals[i+1][0]):
                x = breaks[i+1]
        if x is None and fix_for_cdf:
            m = min(min(v) for v in self.getSegVals())
            M = max(max(v) for v in self.getSegVals())
            if 0 <= y < m:
                x = -inf
            if 1 >= y > M:
                x = inf
        if x is None: # It means
            print("ASSERT x is None y=", y, self.__str__())
            print("ASSERT x is None vals=", vals)
            x = NaN
        return x
    def inverse(self, y):
        if isscalar(y):
            return self.inverse_scalar(y)
        y = array(y)
        x = zeros(size(y))
        for i in range(len(x)):
            x[i] = self.inverse_scalar(y[i])
        return x #findinv(self, a = self.breaks[0], b = self.breaks[-1]-1e-10, c = level, rtol = params.segments.rtol)

    def invfun(self, use_end_poles = True, use_interpolated=True, rangeY=[0,1], include_zero_break=True):
        """
        return inverse of cumulative distribution function as piecewise function
        """
        breakvals = []
        rpoles = []
        lpoles = []
        # only for distribution
        if rangeY is None:
            vals = self.getSegVals()
            for i in range(len(vals)):
                breakvals.extend(vals[i])
            breakvals[0] = vals[0][0]
            breakvals[-1] = vals[-1][1]
            x0 = self(0)
            if x0 not in breakvals:
                breakvals.append(x0)
                breakvals.sort()
        else:
            #breakvals[0] = rangeY[0]
            #breakvals[-1] = rangeY[1]
            breakvals=rangeY
#        if vals[0]>=0 and vals[0]<1:
#            breakvals[0] = 0
#        if vals[-1]>0 and vals[-1]<=1:
#            breakvals[-1] = 1
        def _tmp_inv(x):
            y = self.inverse(x)
            return y
        breakvals = epsunique(array(breakvals),0.0001)
        if include_zero_break:
            if 0 not in breakvals:
                breakvals = numpy.insert(breakvals, numpy.searchsorted(breakvals, 0), 0)
        for i in breakvals:
            lpoles.append(False)
            rpoles.append(False)
        if use_end_poles:
            rpoles[-1]=True
            lpoles[0]=True
        fun2 = PiecewiseFunction(fun = _tmp_inv, breakPoints=breakvals,
                                 lpoles=lpoles, rpoles=rpoles)
        if use_interpolated:
            fun2 = fun2.toInterpolated()
        return fun2

class PiecewiseDistribution(PiecewiseFunction):
    """
    Base representation of probability distribution function using
    piecewise function."""
    def mean(self):
        I = 0.0
        E = 0.0
        for seg in self.segments:
            if seg.isDirac():
                i, e = seg.f*seg.a, 0
            else:
                i, e = _segint(lambda x: x * seg(x), seg.a, seg.b, force_poleL = seg.hasLeftPole(), force_poleU = seg.hasRightPole())
            E += e
            I = I + i
        if E>1.0e-1:
            return NaN
        return I
    def meanf(self, f=None):
        """it gives mean value of f(X) (i.e. E(f(X)))
        """
        if f is None:
            def f(x):
                return x
        I, absI = 0.0, 0.0
        E = 0.0
        for seg in self.segments:
            if seg.isDirac():
                i, e = nan_to_num(f(seg.a))*seg.f, 0
            else:
                i, e = _segint(lambda x: nan_to_num(f(x)) * seg(x), seg.a, seg.b, force_poleL = seg.hasLeftPole(), force_poleU = seg.hasRightPole())
            E += abs(e)
            I = I + i
            absI += abs(i)
        if  abs(E/absI)>1.0e-3 and abs(absI)>1e-10:
            #print I, absI, E
            return NaN
        return I

    def median(self):
        cpf = self.cumint()
        return cpf.inverse(0.5)
    def var(self):
        m = self.mean()
        I = 0.0
        E = 0.0
        for seg in self.getSegments():
            i,e = _segint(lambda x: (x - m) ** 2  * seg(x), seg.a, seg.b, force_poleL = seg.hasLeftPole(), force_poleU = seg.hasRightPole())
            E += e
            I += i
        for seg in self.getDiracs():
            i = seg.f*(seg.a - m) ** 2
            I += i
        if E>1.0e-0:
            return Inf
        else:
            return I

    def std(self):
        return sqrt(self.var())
    def meanad(self):
        """mean absolute deviance"""
        #TODO
        return None
    def medianad(self):
        """median absolute deviance"""
        median = self.median()
        f1 = self.copyShiftedAndScaled(shift = -median, scale = 1.0)
        f2 = f1.splitByPoints([0.0])
        f3 = f2.copyAbsComposition()
        return f3.median()
    def mode(self):
        """Mode, using scipy's function fminbound, may be inaccurate """
        if not have_Scipy_optimize:
            print("Warning: scipy's fminbound not found")
            return None
        m = 0
        x = None
        for seg in self.segments:
            if not seg.isDirac() :
                if seg.hasLeftPole():
                    xi = fminbound(lambda x: -seg(x) + 1e-14, seg.a, seg.b, xtol = 1e-16)
                elif seg.hasRightPole() :
                    xi = fminbound(lambda x: -seg(x), seg.a + 1e-14, seg.b, xtol = 1e-16)
                else:
                    xi = fminbound(lambda x: -seg(x), seg.a, seg.b, xtol = 1e-16)
                mi = float(seg.f(xi))
            else:
                xi, mi = seg.a, seg.f
            if m < mi:
                m = mi
                x = xi
        return x
    def range(self):
        breaks = self.getBreaks()
        return (breaks[0], breaks[-1])
    def iqrange(self, level):
        """Interquartile range for a given level."""
        cpf = self.cumint()
        return cpf.inverse(1-level)-cpf.inverse(level)
    def entropy(self):
        fun = self * self._log1()
        return fun.integrate()

    def L2_distance(self, other):
        fun = (self - other)*(self - other)
        return fun.integrate()**0.5

    def KL_distance(self, other):
        fun = self * (self._log() - other._log())
        return fun.integrate()

    def tailexp(self):
        segMInf = self.segments[0]
        segPInf = self.segments[-1]
        if segMInf.isMInf():
            mexp = segMInf.tailexp()
        else:
            mexp = None
        if segPInf.isPInf():
            pexp = segPInf.tailexp()
        else:
            pexp = None
        return (mexp, pexp)


class CumulativePiecewiseFunction(PiecewiseFunction):
    """
    it represent cumulative intntegral \int_{-\infty}^x f(t) dt
    where f is a piecewise function
    """
    def __init__(self, segments = []):
        PiecewiseFunction.__init__(self, segments)

    def toInterpolated(self):
        interpolatedPFun = CumulativePiecewiseFunction([])
        #for seg in self.segments:
        #    interpolatedPFun.addSegment(seg.toInterpolatedSegment())

        for seg in self.segments:
            if seg.isMInf() and seg.f(seg.b)==seg.f(2*seg.b):           #
                interpolatedPFun.addSegment(seg)                        # TODO: to remove
            elif seg.isPInf() and seg.f(seg.b)==seg.f(2*seg.b):         # add ConstMInfSegment and
                interpolatedPFun.addSegment(seg)                        # ConstPInfSegment instead
            else:                                                       #
                interpolatedPFun.addSegment(seg.toInterpolatedSegment())
        return interpolatedPFun

#    def inverse(self, y):
#        vals = self.getSegVals()
#        breaks = self.getBreaks()
#        x = None
#        if isscalar(y):
#            # coinituaous part of cumilative function
#            for i in range(len(vals)):
#                segi = self.segments[i]
#                if (vals[i][0]<=y<=vals[i][1]):
#                    if segi.isMInf():
#                        x = findinv(segi.f, a = segi.findLeftEps(), b = segi.b, c = y, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter)
#                    elif segi.isPInf():
#                        x = findinv(segi.f, a = segi.a, b = segi.findRightEps(), c = y, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter)
#                    else:
#                        x = findinv(segi.f,  a = segi.a, b = segi.b, c = y, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter) # TODO PInd, MInf
#            # discrete part of cumilative function
#            for i in range(len(vals)-1):
#                if (vals[i][1]<=y) & (y<=vals[i+1][0]):
#                    x = breaks[i+1]
#        else:
#            y = array(y)
#            x = zeros(size(y))
#            # coinituaous part of cumilative function
#            for i in range(len(vals)):
#                segi = self.segments[i]
#                ind = where((vals[i][0]<=y) & (y<=vals[i][1]))
#                if segi.isMInf():
#                    x[ind] = [findinv(segi.f, a = segi.findLeftEps(), b = segi.b, c = yj, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter) for yj in y[ind]]
#                elif segi.isPInf():
#                    x[ind] = [findinv(segi.f, a = segi.a, b = segi.findRightEps(), c = yj, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter) for yj in y[ind]]
#                else:
#                    x[ind] = [findinv(segi.f, a = segi.a, b = segi.b, c = yj, rtol = params.segments.cumint.reltol, maxiter = params.segments.cumint.maxiter) for yj in y[ind]]
#            # discrete part of cumilative function
#            for i in range(len(vals)-1):
#                ind = where((vals[i][1]<=y) & (y<=vals[i+1][0]))
#                x[ind] = breaks[i+1]
#        if (x is None): # It means
#            print "ASSERT x is None y=", y, self.__str__()
#            print "ASSERT x is None vals=", vals
#            assert(False)
#        return x #findinv(self, a = self.breaks[0], b = self.breaks[-1]-1e-10, c = level, rtol = params.segments.rtol)
    def rand(self, n = None, cache = None):
        """Generates random numbers using inverse cumulative distribution function.

        if n is None, return a scalar, otherwise, an array of given
        size."""
        y = uniform(0, 1, n)
        return self.inverse(y)
    def getMaxValue(self, x):
        return self.segments[-1].find

def _findSegListAdd(f, g, z):
    """It find list of segments for integration purposes, for given z
    input: f, g - picewise function, z = x + y
    output: list of segment products depends on z
    """
    seg_list = []
    for segi in f.segments:
        for segj in g.segments:
            if z - segi.a > segj.a and z - segi.b < segj.b:
                seg_list.append((segi, segj))
    return seg_list


#_revoved_to_indeparith
def _segint(fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
            debug_info = False, debug_plot = False):
    #print params.integration_infinite.exponent
    if L > U:
        if params.segments.debug_info:
            print("Warning: reversed integration interval, returning 0")
        return 0, 0
    if L == U:
        return 0, 0
    if force_minf:
        #i, e = integrate_fejer2_minf(fun, U, a = L, debug_info = debug_info, debug_plot = True)
        i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
    elif force_pinf:
        #i, e = integrate_fejer2_pinf(fun, L, b = U, debug_info = debug_info, debug_plot = debug_plot)
        i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
    elif not isinf(L) and  not isinf(U):
        if force_poleL and force_poleU:
            i1, e1 = integrate_fejer2_Xn_transformP(fun, L, (L+U)*0.5, debug_info = debug_info, debug_plot = debug_plot)
            i2, e2 = integrate_fejer2_Xn_transformN(fun, (L+U)*0.5, U, debug_info = debug_info, debug_plot = debug_plot)
            i, e = i1+i2, e1+e2
        elif force_poleL:
            i, e = integrate_fejer2_Xn_transformP(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        elif force_poleU:
            i, e = integrate_fejer2_Xn_transformN(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        else:
            #i, e = integrate_fejer2(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
            i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
    elif isinf(L) and isfinite(U) :
        #i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        i, e = integrate_fejer2_minf(fun, U, debug_info = debug_info, debug_plot = debug_plot, exponent = params.integration_infinite.exponent,)
    elif isfinite(L) and isinf(U) :
        #i, e = integrate_wide_interval(fun, L, U, debug_info = debug_info, debug_plot = debug_plot)
        i, e = integrate_fejer2_pinf(fun, L, debug_info = debug_info, debug_plot = debug_plot, exponent = params.integration_infinite.exponent,)
    elif L<U:
        i, e = integrate_fejer2_pminf(fun, debug_info = debug_info, debug_plot = debug_plot, exponent = params.integration_infinite.exponent,)
    else:
        print("errors in _conv_div: x, segi, segj, L, U =", L, U)
    #print "========"
    #if e>1e-10:
    #    print "error L=", L, "U=", U, fun(array([U])), force_minf , force_pinf , force_poleL, force_poleU
    #    print i,e
    return i,e


def _conv_diracs(f, g, fun = operator.add):
    """discrete convolution of f and g
    """
    fg = PiecewiseFunction([])
    wyn = {}
    for fi in f.getDiracs():
        for gi in g.getDiracs():
            key = fun(fi.a, gi.a)
            if key in wyn:
                wyn[key] = wyn.get(key) + fi.f * gi.f
            else:
                wyn[key] = fi.f * gi.f
    for key in list(wyn.keys()):
        fg.addSegment(DiracSegment(key, wyn.get(key)))
    return fg

def _makeCommonSegmentList(f, g):
    allbreaks = union1d(f.getBreaks(), g.getBreaks())
    commonList = []
    for i in range(len(allbreaks)-1):
        a = allbreaks[i]
        b = allbreaks[i+1]
        seg1 = f.findSegmentByEnds(a, b)
        seg2 = g.findSegmentByEnds(a, b)
        commonList.append((seg1, seg2))
    return commonList


def _safe_call(x):
    if isscalar(x):
        if isnan(x):
            return 0.0
        else:
            return x
    else:
        mask = (isnan(x))
        x[mask] = 0
        return x


if __name__ == "__main__":
    import time
#    from pylab import show, subplot
#    import matplotlib.pyplot as plt
#    ts = time.time()
#    """Product of N mixtures of uniform random variables"""
#    segf1 = DiracSegment(1.0, 0.3)
#    segf2 = DiracSegment(0.5, 0.5)
#    segf3 = Segment(0.0, 1.0, lambda x: 6.0*x*(1.0-x))
#    segf4 = Segment(0.0, 1.0, lambda x: 1-abs(x-1))
#    segf5 = Segment(1.0, 2.0, lambda x: 1-abs(x-1))
#    segf6 = ConstSegment(-1.0, -0.5, 2.0/1.0)
#    segf7 = ConstSegment(-2.0, -0.0, 1.0/3.0)
#    segf8 = ConstSegment(0.0, 1.0, 3.0/3.0)
#    segf9 = ConstSegment(2.0, 3.0, 1.0/2.0)
#    #seglog = SegmentWithPole(0, 0.5, lambda x: log(x)**4, pole = 0.0)
#    #seglog2 = Segment(0.5, 1, lambda x: log(x)**4)
#    seglog = SegmentWithPole(0.0, 1.0, lambda x: 0.5/x**0.5)
#    #seglog = SegmentWithPole(0.0, 1.0, lambda x: 0.25/abs(x)**0.5, pole = 0.0)
#    #seglog1 = SegmentWithPole(-1.0, 0.0, lambda x: 0.25/abs(x)**0.5, pole = 0.0)
#    #seglog1 = Segment(1.0, 10, lambda x: 0.5/x**0.5)
#    #seglog2 = Segment(1.0, 10, lambda x: 0.5/x**0.5)
#    #seglog2 = SegmentWithPole(-1.0, 0.0, lambda x: 0.5/abs(x)**0.5, pole = 0.0)
#    #seglog = SegmentWithPole(0, 1, lambda x: chisqr(x,1), pole = 0.0)
#    #seglog2 = PInfSegment(1, lambda x: chisqr(x,1))
#    #segg1 = MInfSegment(-1.0, lambda x:normpdf(x))
#    segg1 = Segment(-3.0, -1.0, lambda x:normpdf(x))
#    segg2 = PInfSegment(1.0, lambda x:normpdf(x))
#    segg3 = Segment(-1.0, 0.0, lambda x:normpdf(x))
#    segg4 = Segment(+0.0, 1.0, lambda x:normpdf(x))
#
#    seg1 =  Segment( 0.0, 1.0,  lambda x: 0.5 + 0.0*x)
#    seg2 =  Segment(-1.0, 0.0,  lambda x: 0.5 + 0.0*x)
#    f = PiecewiseFunction([])
#    seglog = SegmentWithPole(0.0, 1.0, lambda x: 0.5/abs(x)**0.5)
#    seglog2 = SegmentWithPole(-1.0, 0.0, lambda x: 0.5/abs(x)**0.5, left_pole = False)
#    #f.addSegment(segf3)
#    #f.addSegment(seglog2)
#    #g1 = f.toInterpolated()
#    #f.addSegment(seglog2)
#    #c = f.copySquareComposition()
#    #d = c.toInterpolated()
#    # f.addSegment(seg1)
#
#    #g.addSegment(segg1)
#    #g.addSegment(segg2)
#    #g.addSegment(segg3)
#    #g.addSegment(segg4)
#    #print f.medianad()
#    ##f.plot(linewidth=1, color = 'g', linestyle='-')
#    ##g1.plot(linewidth=1, color = 'b', linestyle='-')
#    ##g.plot(linewidth=3, color = 'r', linestyle='-')
#    #f.plot()
#    #g = f.copyShiftedAndScaled(4,0.6)
#    #g.plot(color = 'r')
#    #h= _conv_mean(f,g, p=0.5,q=0.5)
#    #h.plot(color = 'k')
#    #print g.integrate(), g
#    #print h.integrate(), h
#    #print h.medianad()
#    #fig = plt.figure()
#
#    #f = PiecewiseFunction([])
#    #f.addSegment(SegmentWithPole(0,1,lambda x: 1/x))
#
#    #f.addSegment(PInfSegment(1,lambda x: exp(-x)))
#    #f.addSegment(MInfSegment(-1,lambda x: exp(x)))
#    #f.addSegment(Segment(-1, 0,lambda x: 1+x))
#    #print "f=",f
#    #g = f.splitByPoints(array([-7, -4, -0.3, 0.5, 0.8, 1.3, 5.0]))
#    #print g
#    #f.plot(color = "k")
#    #g.plot()
#    #h = g.toInterpolated()
#    #print h
#    #h.plot(color = 'r')
#    #r=h-f
#    #plt.figure()
#    #r.plot()
#    f = PiecewiseFunction(fun = lambda x : 1.0/abs(x)**0.5, breakPoints = [-5, -3,-2, 0,1, 10], lpoles=[False, False, False, True, False, False],rpoles=[False, False, False, True, False, False])
#    f.addSegment(DiracSegment(1,0.7))
#    plt.figure()
#    f.plot(show_segments= True)
#    #plt.figure()
#    #g  = f.toInterpolated()
#    #g.plot(show_segments= True, show_nodes=True)
#    #plt.figure()
#    #g.plot(xmin =-10, xmax = 5, show_segments= True, show_nodes=True, color = 'k', linewidth= 1)
#    #plt.figure()
#    #g.plot(xmin =-20, xmax = 10,  show_segments= False, show_nodes=True, color = 'k', linewidth= 1)
#    #plt.figure()
#    #r = f-g
#    #r.plot(xmin =-20, xmax = 10,  show_segments= False, show_nodes=True, color = 'k', linewidth= 1)
#    plt.figure()
#    k = PiecewiseDistribution([])
#    segf1 = DiracSegment(0.0, 0.3)
#    segf2 = DiracSegment(0.5, 0.5)
#    segf3 = Segment(0.0, 0.5, lambda x: 0.8*x)
#    segf4 = ConstSegment(0.5, 1.0, 0.2)
#    k.addSegment(segf1)
#    k.addSegment(segf2)
#    k.addSegment(segf3)
#    k.addSegment(segf4)
#    print k
#
#
#    k.plot()
#    show()
#    0/0
#    #h = k.toInterpolated()
#    from indeparith import conv, convmax, convmin
#    i  = conv(h,h)
#    print "h=", h.integrate(), h
#    print "i=", i.integrate(), i
#
#
#    h.plot()
#    i.plot()
#    plt.figure()
#    c = g.cumint().toInterpolated()
#    print c
#    c.plot()
#    plt.figure()
#    f = PiecewiseFunction(fun  = exp, breakPoints = [-10, -7, 0, 3])
#    g = PiecewiseFunction(fun  = sin, breakPoints  = [-11, -7, 1, 3])
#    f = f.toInterpolated()
#    g = g.toInterpolated()
#
#    h= f-g
#    h.plot(color = 'r', linewidth=3.0)
#    f.plot(color = 'b')
#    g.plot(color = 'k')
#
#    print h
#    print h.segments[1].f(linspace(-1,2,17))
#    print f(linspace(-1,2,100)) - g(linspace(-1,2,100))
#    print g.segments[1].f(linspace(-1,2,17))
#    show()
#
#    h = PiecewiseDistribution([])
#    k.addSegment(ConstSegment(1.0,2.0,1.0/2.0))
#    #k.addSegment(Segment(0.0, 0.5,lambda x: 1.0+0.0*x))
#    h.addSegment(Segment(-1.0,3.0,lambda x: 1.0/3.0+0.0*x))
#    #k.addSegment(Segment(0.2,1.0,lambda x: 1.0+0.0*x))
#    #k = k.toInterpolated()
#    #h.addSegment(ConstSegment(-1,0,0.5))
#    print k, k.range()
#
#    f.plot(show_segments = True)
#    print "f=", f.integrate(0,1), f
#    k.plot()
#    h.plot()
#    p = convmin(k,h)
#    print "======", p
#    plt.figure()
#    p.plot()
#    plt.figure()
#    p.plot(xmin =-10, xmax = 5, leftRightEpsilon = 1e-2)
#    plt.figure()
#    p.plot(xmin =-1, xmax = 2, leftRightEpsilon = 1e-2)
#
#
#    #g = _conv_(g1,g1)
#    #g.plot(linewidth=4, color = 'g', linestyle='-')
#    #intf =  f.integrate()
#    #intg1 =  g1.integrate()
#    #intg =  g.integrate()
#
#    #print 'initegral=', 1.0-intf,1.0-intg1,1.0-intg
#    #print 'initegral=', intf,intg1,intg2, intf**1-intg1, intf**2-intg2
#    #dd = _conv_(d,d)
#    #dd.plot()
#
#    #plt.figure()
#    #f.plot_tails()
#    #g.plot_tails()
#
#
#
#
#    #d.plot()
#    #fig = plt.figure()
#    #print estimateDegreeOfZero(h, Inf)
#
#    #intf =  f.integrate()
#    #intg =  g.integrate()
#    #print intf
#    #print intg
#    #print intf - intg
#    #plt.figure()
#    #f.plot_tails()
#    #plt.figure()
#    #h.plot_tails()
#    plt.show()
