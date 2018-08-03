#    PaCal - the probabilistic calculator
#    Copyright (C) 2009  Szymon Jaroszewicz
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

from functools import partial

from numpy import array, asfarray, zeros_like, ones_like, asarray, atleast_1d
from numpy import array_split, concatenate, squeeze
from numpy import where, dot, zeros
from numpy import maximum
from numpy import newaxis, subtract, isscalar, hstack
from numpy import finfo, size, double
from numpy import isnan, isinf
from numpy import arange, cos, sin, log, exp, pi, log1p, expm1

from numpy import cumsum, flipud, real, imag, linspace
from numpy.linalg import eigvals
#from numpy.polynomial.chebyshev import chebroots
from . import params

from .utils import cheb_nodes_log, incremental_cheb_nodes_log
from .utils import cheb_nodes, incremental_cheb_nodes, cheb_nodes1, incremental_cheb_nodes1
from .utils import combine_interpolation_nodes, combine_interpolation_nodes_fast
from .utils import convergence_monitor, chebspace, chebspace1, estimateDegreeOfPole
from .utils import debug_plot
from .utils import chebt2, ichebt2, chebt1, ichebt1, chebroots

from .vartransforms import *


# import faster Cython versions if possible
try:
    #import pyximport; pyximport.install()
    from .bary_interp import bary_interp
    have_Cython = True
    print("Using compiled interpolation routine")
except:
    print("Compiled interpolation routine not available")
    have_Cython = False


class Interpolator(object):
    """Polynomial interpolation base class"""
    def __init__(self, Xs, Ys):
        self.Xs = array(Xs)
        self.Ys = array(Ys)
        #self.n = len(self.Xs)
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes."""
        self.Xs, self.Ys = combine_interpolation_nodes(self.Xs, self.Ys, new_Xs, new_Ys)
    def interp_at(self, x):
        """Simple Lagrange interpolation.
        Slow."""
        ret = 0
        for i, xi in enumerate(self.Xs):
            weight = 1.0
            for j, xj in enumerate(self.Xs):
                if j != i:
                    weight *= (x - xj) / (xi - xj)
            ret += weight * self.Ys[i]
        return ret
    def __call__(self, x):
        return self.interp_at(x)
    def getNodes(self):
        return self.Xs, self.Ys
    def copyReverse(self):
        return Interpolator(self.Xs[-1:1:-1], self.Ys[-1:1:-1])

class BarycentricInterpolator(Interpolator):
    """Barycentric interpolation"""
    def __init__(self, Xs, Ys, weights=None):
        super(BarycentricInterpolator, self).__init__(Xs, Ys)
        if (weights is None):
            self.init_weights(Xs)
        else:
            self.weights = weights
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes."""
        super(BarycentricInterpolator, self).add_nodes(new_Xs, new_Ys)
        self.init_weights(self.Xs)
    def init_weights(self, Xs):
        """Using Lagrange method"""
        weights = []
        for i, xi in enumerate(self.Xs):
            weight = 1.0
            for j, xj in enumerate(self.Xs):
                if j != i:
                    weight /= (xi - xj)
            weights.append(weight)
        self.weights = array(weights)
    def interp_at(self, x):
        """Barycentric interpolation"""
        scalar_x = isscalar(x)
        if have_Cython:
            y = bary_interp(self.Xs, self.Ys, self.weights, asfarray(atleast_1d(x)))
            if scalar_x:
                y = y[0]
        else:
            # split x if necessary to avoid overflow
            max_size = 100000
            if scalar_x:
                x_parts = [atleast_1d(x)]
            elif x.size <= max_size:
                x_parts = [x]
            else:
                n_parts = (x.size * len(self.Xs) + max_size - 1) // max_size
                x_parts = array_split(x, n_parts)
            # now interpolate each part of x
            results = []
            for x_p in x_parts:
                xdiff = subtract.outer(x_p, self.Xs)
                ind = where(xdiff == 0)
                xdiff[ind] = 1
                temp = self.weights / xdiff
                num = dot(temp, self.Ys)
                den = temp.sum(axis= -1)
                # Tricky case which can occur when ends of the interval are
                # almost equal.  xdiff can be close to but nonzero, but the sum
                # in the denominator can be exactly zero.
                if (den == 0).any():
                    num[den == 0] = self.Ys[abs(xdiff[den == 0]).argmin(axis= -1)]
                    den[den == 0] = 1
                ret = array(num / den)
                if len(ind[0]) > 0:
                    ret[ind[:-1]] = self.Ys[ind[-1]]
                results.append(ret)
            # concatenate results
            if scalar_x:
                y = squeeze(results)
            else:
                y = concatenate(results)
        return y
    def copyShiftedAndScaled(self, shift, scale):
        return BarycentricInterpolator((self.Xs[::-1] - shift) / scale, self.Ys[::-1] * scale, self.weights[::-1])
    def diff(self, use_2nd=True):
        if use_2nd:
            c = chebt2(self.Ys)
            n = len(c);
            cdiff = zeros(n + 1);
            v = concatenate(([0, 0], 2 * arange(n - 1, 0, -1) * c[0:-1]));
            cdiff[0::2] = cumsum(v[0::2]);
            cdiff[1::2] = cumsum(v[1::2]);
            cdiff[-1] = .5 * cdiff[-1];
            cdiff = cdiff[2:];
            Ydiffvals = ichebt2(cdiff) / ((self.Xs[-1] - self.Xs[0]) * 0.5)
            Ydiffvals = flipud(Ydiffvals)
            Xs, Ws = chebspace(self.Xs[0], self.Xs[-1], len(Ydiffvals), returnWeights=True)
            f = BarycentricInterpolator(Xs, Ydiffvals, Ws)
            return f
        else:
            Xs, Ws = chebspace(self.Xs[0], self.Xs[-1], len(self.Xs)+2, returnWeights=True)
            Ys = self.interp_at(Xs)
            return BarycentricInterpolator(Xs, Ys, Ws).diff(use_2nd=True)

    def roots(self, use_2nd=True):
        if use_2nd:
            cs = flipud(chebt2(self.Ys))
            roots = chebroots(cs)
            reals = abs(imag(roots)) < params.interpolation.convergence.abstol
            roots = real(roots[reals])
            unit = (roots>-1.0) & (roots < 1.0)
            roots = roots[unit]
            return (roots*(self.Xs[-1]-self.Xs[0])+(self.Xs[0]+self.Xs[-1]))*0.5
        else:
            Ys = self.interp_at(chebspace(self.Xs[0], self.Xs[-1], self.n))
            cs = flipud(chebt2(Ys))
            roots = chebroots(cs)
            reals = abs(imag(roots)) < params.interpolation.convergence.abstol
            roots = real(roots[reals])
            unit = (roots>-1.0) & (roots < 1.0)
            roots = roots[unit]
            return (roots*(self.Xs[-1]-self.Xs[0])+(self.Xs[0]+self.Xs[-1]))*0.5
    def trim(self, abstol=None):
        return self
class AdaptiveInterpolator(object):
    """Mix-in class for adaptive interpolators.

    Increase number of nodes until error is small."""
    def adaptive_init(self, f, interp_class):
        if f is not None:
            self.f = f
        self.interp_class = interp_class
        self.n = 3
        Xs = self.get_nodes(self.n)
        Ys = self.f(Xs)
        interp_class.__init__(self, Xs, Ys)
    def adaptive_interp(self, par=None):
        if par is None:
            par = params.interpolation
        maxn = par.maxn
        n = self.n
        old_err = None
        cm = convergence_monitor(par=par.convergence)
        while n <= maxn:
            new_n = 2 * n - 1
            new_Xs = self.get_incremental_nodes(new_n)
            new_Ys = self.f(new_Xs)
            err = self.test_accuracy(new_Xs, new_Ys)
            maxy = max(abs(new_Ys).max(), abs(self.Ys).max())
            if par.debug_info:
                print("interp. err", err, old_err, new_n)
            cm.add(err, maxy)
            if cm.test_convergence()[0]:
                break
            old_err = err
            n = new_n
            self.add_nodes(new_Xs, new_Ys)
        self.n = n
        if par.debug_plot and n >= maxn:
            debug_plot(self.a, self.b, self.Xs, self.Ys, None)
        if par.debug_info:
            print("interp. err = ", err, "nodes=", n)
        self.err = err
    def test_accuracy(self, new_Xs, new_Ys):
        """Test accuracy by compa true and interpolated values at
        given points."""
        errs = abs(self.interp_class.interp_at(self, new_Xs) - new_Ys)
        err = errs.max()
        return err

class ChebyshevInterpolator(BarycentricInterpolator, AdaptiveInterpolator):
    """Adaptive Chebyshev interpolator"""
    def __init__(self, f, a, b, *args, **kwargs):
        self.a = float(a)
        self.b = float(b)
        self.adaptive_init(f, BarycentricInterpolator)
        self.adaptive_interp(*args, **kwargs)
    def init_weights(self, Xs):
        self.weights = ones_like(Xs)
        self.weights[::2] = -1
        self.weights[0] /= 2
        self.weights[-1] /= 2
    def get_nodes(self, n):
        return cheb_nodes(n, self.a, self.b)
    def get_incremental_nodes(self, new_n):
        return incremental_cheb_nodes(new_n, self.a, self.b)
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes. A faster version assuming nesting is possible here."""
        self.Xs, self.Ys = combine_interpolation_nodes_fast(self.Xs, self.Ys, new_Xs, new_Ys)
        self.init_weights(self.Xs)
    def trim(self, abstol=None):
        c = chebt2(self.Ys)
        if abstol is None:
            abstol = params.interpolation.convergence.abstol
        while c[0] < abstol:
            c = c[1:]
        Ydiffvals = ichebt2(c)
        Ydiffvals = flipud(Ydiffvals)
        Xs, Ws = chebspace(self.a, self.b, len(Ydiffvals), returnWeights=True)
        f = BarycentricInterpolator(Xs, Ydiffvals, Ws)
        return f
    def diff(self, use_2nd=True):
        return super(ChebyshevInterpolator, self).diff(use_2nd=use_2nd)
    def roots(self, use_2nd=True):
        return super(ChebyshevInterpolator, self).roots(use_2nd=use_2nd)

class LogXChebyshevInterpolator(BarycentricInterpolator, AdaptiveInterpolator):
    """Adaptive Chebyshev interpolator"""
    def __init__(self, f, a, b, *args, **kwargs):
        self.a = float(a)
        self.b = float(b)
        self.adaptive_init(f, BarycentricInterpolator)
        self.adaptive_interp(*args, **kwargs)
    def init_weights(self, Xs):
        self.weights = ones_like(Xs)
        self.weights[::2] = -1
        self.weights[0] /= 2
        self.weights[-1] /= 2
    def get_nodes(self, n):
        return cheb_nodes_log(n, self.a, self.b)
    def get_incremental_nodes(self, new_n):
        return incremental_cheb_nodes_log(new_n, self.a, self.b)
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes. A faster version assuming nesting is possible here."""
        self.Xs, self.Ys = combine_interpolation_nodes_fast(self.Xs, self.Ys, new_Xs, new_Ys)
        self.init_weights(self.Xs)

class ChebyshevInterpolatorNoL(ChebyshevInterpolator):
    """Adaptive Chebyshev interpolator"""
    def init_weights(self, Xs):
        self.weights = ones_like(Xs)
        self.weights[::2] = -1
        self.weights[0] /= 2
        n = len(Xs) + 1
        #self.weights *= 1 + cheb_nodes(n)[-1:0:-1]
        # more stable version of the above modification:
        self.weights *= 2 * (sin(arange(n) * pi / (n - 1) / 2)[-1:0:-1]) ** 2
        self.weights = self.weights[::-1]
    def get_nodes(self, n):
        return cheb_nodes(n, self.a, self.b)[1:]
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes. A faster version assuming nesting is possible here."""
        self.Xs, self.Ys = combine_interpolation_nodes_fast(new_Xs, new_Ys, self.Xs, self.Ys)
        self.init_weights(self.Xs)
    def diff(self):
        return super(ChebyshevInterpolatorNoL, self).diff(use_2nd=False)
    def roots(self):
        return super(ChebyshevInterpolatorNoL, self).roots(use_2nd=False)
class ChebyshevInterpolatorNoR(ChebyshevInterpolator):
    """Adaptive Chebyshev interpolator without rightmost point"""
    def init_weights(self, Xs):
        self.weights = ones_like(Xs)
        self.weights[1::2] = -1
        self.weights[-1] /= 2
        n = len(Xs) + 1
        #self.weights *= cheb_nodes(n)[-2::-1] - 1
        # more stable version of the above modification:
        self.weights *= 2 * sin(pi / 2 - arange(n) * pi / (n - 1) / 2)[-2::-1] ** 2
        self.weights = self.weights[::-1]
    def get_nodes(self, n):
        return cheb_nodes(n, self.a, self.b)[:-1]
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes. A faster version assuming nesting is possible here."""
        self.Xs, self.Ys = combine_interpolation_nodes_fast(self.Xs, self.Ys, new_Xs, new_Ys)
        self.init_weights(self.Xs)
    def diff(self):
        return super(ChebyshevInterpolatorNoR, self).diff(use_2nd=False)
    def roots(self):
        return super(ChebyshevInterpolatorNoR, self).roots(use_2nd=False)

class AdaptiveInterpolator1(object):
    """Mix-in class for adaptive interpolators with 3*n rule"""
    def adaptive_init(self, f, interp_class):
        self.f = f
        self.interp_class = interp_class
        n = 2
        Xs = self.get_nodes(n)
        Ys = self.f(Xs)
        interp_class.__init__(self, Xs, Ys)
    def adaptive_interp(self, par=None):
        if par is None:
            par = params.interpolation
        # increase number of nodes until error is small
        maxn = par.maxn
        n = len(self.Xs)
        old_err = None
        cm = convergence_monitor(par=par.convergence)
        while n <= maxn:
            new_n = 3 * n
            new_Xs = self.get_incremental_nodes1(new_n)
            new_Ys = self.f(new_Xs)
            err = self.test_accuracy(new_Xs, new_Ys)
            maxy = max(abs(new_Ys).max(), abs(self.Ys).max())
            if par.debug_info:
                print("interp. err1", err, maxy, old_err, "nodes=", n, maxn)
            cm.add(err, maxy)
            if cm.test_convergence()[0]:
                break
            old_err = err
            n = new_n
            self.add_nodes(new_Xs, new_Ys)
        self.n = n
        if par.debug_plot and n >= maxn:
            debug_plot(self.a, self.b, self.Xs, self.Ys, None)
        if par.debug_info:
            print("interp. err1 = ", err, "nodes=", n)
        self.err = err
    def test_accuracy(self, new_Xs, new_Ys):
        """Test accuracy by comparing true and interpolated values at
        given points."""
        errs = abs(self.interp_class.interp_at(self, new_Xs) - new_Ys)
        err = errs.max()
        return err

class ChebyshevInterpolator1(BarycentricInterpolator, AdaptiveInterpolator1):
    """Adaptive Chebyshev interpolator based on nodes of 1-st kind"""
    def __init__(self, f, a, b, **adapt_args):
        self.a = float(a)
        self.b = float(b)
        self.adaptive_init(f, BarycentricInterpolator)
        self.adaptive_interp(**adapt_args)
    def init_weights(self, Xs):
        self.weights = ones_like(Xs)
        n = len(self.weights)
        self.weights = sin(arange(1, 2 * n, 2) * pi / (2 * n))
        self.weights[1::2] = -1 * self.weights[1::2]
    def get_nodes(self, n):
        return cheb_nodes1(n, self.a, self.b)
    def get_incremental_nodes1(self, new_n):
        return incremental_cheb_nodes1(new_n, self.a, self.b)
    def add_nodes(self, new_Xs, new_Ys):
        """Add new interpolation nodes. A faster version assuming nesting is possible here."""
        self.Xs, self.Ys = combine_interpolation_nodes(self.Xs, self.Ys, new_Xs, new_Ys)
        self.init_weights(self.Xs)
    def diff(self):
        return super(ChebyshevInterpolator1, self).diff(use_2nd=False)
    def roots(self):
        return super(ChebyshevInterpolator1, self).roots(use_2nd=False)
def _find_zero(f, a, ymax=1e-120, ymin=1e-150):
    """Find where a function achieves very small values (but greater
    than zero)."""
    x_min = 1e-5
    x_max = 1e100
    x_mid = exp((log(x_min) + log(x_max)) / 2)
    try:
        y = f(a + x_mid)
    except OverflowError:
        y = 0
    while x_max > x_min * (1.0 + 1e-5) and (isnan(y) or isinf(y) or y < ymin or y > ymax):
        if isinf(y) or isnan(y) or y < ymin:
            x_max = x_mid
        else:
            x_min = x_mid
        x_mid = exp((log(x_min) + log(x_max)) / 2)
        try:
            y = f(a + x_mid)
        except OverflowError:
            y = 0
    return a + x_mid

def _call_f(interp, x):
    return interp.spec_f(x)
def _wrap_f(f):
    if params.general.parallel:
        return partial(_call_f, f.__self__)
    return f

class ValTransformInterpolator(ChebyshevInterpolator1):
    def __init__(self, f, a, b=None, val_transform=log, val_transform_inv=exp, *args, **kwargs):
        self.wrapped_f = f
        if b is None:
            b = _find_zero(f, a)
        self.val_transform = val_transform
        self.val_transform_inv = val_transform_inv
        super(ValTransformInterpolator, self).__init__(_wrap_f(self.spec_f),
                                                       self.val_transform(a), self.val_transform(b),
                                                       *args, **kwargs)
    def spec_f(self, x):
        return self.val_transform(self.wrapped_f(self.val_transform_inv(x)))
    def interp_at(self, x):
        return self.val_transform_inv(super(ValTransformInterpolator, self).interp_at(self.val_transform(x)))
    def __call__(self, x):
        return self.interp_at(x)
    def get_nodes(self, n):
        return cheb_nodes1(n, self.a, self.b)
    def get_incremental_nodes1(self, new_n):
        return incremental_cheb_nodes1(new_n, self.a, self.b)
    def getNodes(self):
        return self.val_transform(self.Xs), self.val_transform_inv(self.Ys)
    def test_accuracy(self, new_Xs, new_Ys):
        """Test accuracy by comparing true and interpolated values at
        given points."""
        errs = abs(self.val_transform_inv(self.interp_class.interp_at(self, new_Xs)) - self.val_transform_inv(new_Ys))
        err = errs.max()
        return err

### interpolators with better asymptotics around zero

#        self.a = a
#        self.b = b


#class ChebyshevInterpolatorNoL2(ChebyshevInterpolatorNoL):
class ChebyshevInterpolatorNoL2(ChebyshevInterpolator1):
    def __init__(self, f, a, b=None, par=None, *args, **kwargs):
        if par is None:
            par = params.interpolation
        self.exponent = estimateDegreeOfPole(f, a)
        self.a = a
        self.b = b
        if par.debug_info:
            print("exponent=", self.exponent)
        super(ChebyshevInterpolatorNoL2, self).__init__(lambda x: f(x) / x ** self.exponent, a, b,
                                                        par=params.interpolation,
                                                        *args, **kwargs)
    def interp_at(self, x):
        return super(ChebyshevInterpolatorNoL2, self).interp_at(x) * x ** self.exponent
    def getNodes(self):
        return self.Xs, self.Ys * self.Xs ** self.exponent

class ChebyshevInterpolatorNoR2(ChebyshevInterpolator1):
#class LogTransformInterpolator(ChebyshevInterpolator1):
    def __init__(self, f, a, b=None, par=None, *args, **kwargs):
        if par is None:
            par = params.interpolation
        self.exponent = estimateDegreeOfPole(f, a)
        self.a = a
        self.b = b
        if par.debug_info:
            print("exponent=", self.exponent)
        super(ChebyshevInterpolatorNoR2, self).__init__(lambda x: f(x) / abs(x) ** self.exponent, a, b,
                                                       par=params.interpolation,
                                                       *args, **kwargs)
    def interp_at(self, x):
        return super(ChebyshevInterpolatorNoR2, self).interp_at(x) * abs(x) ** self.exponent
    def getNodes(self):
        return self.Xs, self.Ys * self.Xs ** self.exponent

class LogTransformInterpolator(ChebyshevInterpolatorNoR):
#class LogTransformInterpolator(ChebyshevInterpolatorNoR2):
#class LogTransformInterpolator(ChebyshevInterpolator1):
    def xt(self, x):
        return expm1(x) + self.offset
    def xtinv(self, x):
        return log1p(x - self.offset)
    def __init__(self, f, a, b=None, offset=1, par=None, *args, **kwargs):
        if f is not None:
            self.wrapped_f = f
        if par is None:
            par = params.interpolation_asymp
        if b is None:
            b = _find_zero(f, a)
            if par.debug_info:
                print("found", b, f(b))
        self.orig_a = a
        self.orig_b = b
        if a < 0:
            offset = a + offset
        self.offset = offset - 1
        super(LogTransformInterpolator, self).__init__(_wrap_f(self.spec_f),
                                                       self.xtinv(self.orig_a), self.xtinv(self.orig_b),
                                                       par=params.interpolation_asymp,
                                                       *args, **kwargs)
    def spec_f(self, x):
        return log(abs(self.wrapped_f(self.xt(x))))
    def interp_at(self, x):
        if isscalar(x):
            if x > self.orig_b:
                y = 0
            else:
                y = exp(super(LogTransformInterpolator, self).interp_at(self.xtinv(x)))
            return y
        else:
            x = asfarray(x)
            y = zeros_like(x)
            mask = (x <= self.orig_b)
            y[mask] = exp(super(LogTransformInterpolator, self).interp_at(self.xtinv(x[mask])))
        return y
    def getNodes(self):
        xs, ys = super(LogTransformInterpolator, self).getNodes()
        return self.xt(xs), exp(ys)
#from numpy import polyval, polyfit
#class PolyInterpolator(object):
#    """Explicit polynomial interpolation.
#
#    Should work better for small arguments.  Left end of the interval
#    is assumed to be 0."""
#    def __init__(self, f, b, maxdeg = 10):
#        self.f = f
#        self.a = 0
#        self.b = b
#        Xs = asfarray([0])
#        Ys = asfarray([0])
#        poly = [0]
#        d = 3
#        cm = convergence_monitor()
#        while d <= maxdeg:
#            newXs = cheb_nodes(d, 0, self.b)
#            newYs = f(newXs)
#            print newXs, newYs
#            estYs = polyval(poly, newXs)
#            maxy = max(abs(newYs))
#            err = max(abs(newYs - estYs))
#            cm.add(err, maxy)
#            if cm.test_convergence()[0]:
#                break
#            poly = polyfit(newXs, newYs, d)
#            Xs = newXs
#            Ys = newYs
#            d += 1
#        self.Xs = Xs
#        self.Ys = Ys
#        self.d = len(poly) - 1
#        self.poly = poly
#        self.err = err
#    def interp_at(self, x):
#        return polyval(self.poly, x)
#    def __call__(self, x):
#        return self.interp_at(x)
#    def getNodes(self):
#        return self.Xs, self.Ys


#class PoleInterpolatorP(ChebyshevInterpolator1):
class PoleInterpolatorP(ChebyshevInterpolatorNoL):
    def xt(self, x):
        return (exp(x) + self.orig_a) + self.offset
    def xtinv(self, x):
        return log((x - self.orig_a) + self.offset)
    def __init__(self, f, a, b, offset=1e-50, *args, **kwargs):
        if f is not None:
            self.wrapped_f = f
        self.orig_a = a
        self.orig_b = b
        self.sign = int(sign(self.wrapped_f(float(a + b) / 2)))
        if self.sign == 0:
            self.sign = 1
        if a == 0:
            offset = 1e-50
        else:
            offset = abs(a) * finfo(double).eps
        self.offset = offset
        super(PoleInterpolatorP, self).__init__(_wrap_f(self.spec_f),
                                                self.xtinv(self.orig_a), self.xtinv(self.orig_b),
                                                *args, **kwargs)
    def spec_f(self, x):
        return log1p(self.sign * self.wrapped_f(self.xt(x)))
    def interp_at(self, x):
        y = self.sign * expm1(abs(super(PoleInterpolatorP, self).interp_at(self.xtinv(x))))
        return y
    def getNodes(self):
        return self.xt(self.Xs), self.sign * expm1(self.Ys)
    def test_accuracy_tmp(self, new_Xs, new_Ys):
        """Test accuracy by comparing true and interpolated values at
        given points."""
        errs = abs(self.interp_class.interp_at(self, new_Xs) - new_Ys) * self.xt(new_Ys) / max(self.xt(new_Ys))
        err = errs.max()
        return err
    def diff(self):
        Xs, Ws = chebspace(self.xtinv(self.orig_a), self.xtinv(self.orig_b), self.n, returnWeights=True)
        Ys = concatenate(([self.Ys[0]], self.Ys))
        dp = BarycentricInterpolator(Xs, Ys, Ws).diff()
        Xs, Ws = chebspace1(self.orig_a, self.orig_b, self.n, returnWeights=True)
        Ys = exp(super(PoleInterpolatorP, self).interp_at(self.xtinv(Xs))) * dp(self.xtinv(Xs)) /((Xs - self.orig_a) + self.offset)
        return BarycentricInterpolator(Xs=Xs, Ys=Ys, weights=Ws)
    def roots(self):
        return array([])
    def roots_in_diff(self):
        Xs, Ws = chebspace(self.xtinv(self.orig_a), self.xtinv(self.orig_b), self.n, returnWeights=True)
        Ys = concatenate(([self.Ys[0]], self.Ys))
        dp = BarycentricInterpolator(Xs, Ys, Ws).diff()
        return dp.roots()

class PoleInterpolatorN(ChebyshevInterpolatorNoR):
    def xt(self, x):
        return (-exp(x) + self.orig_b) - self.offset
    def xtinv(self, x):
        return log(-(x - self.orig_b) + self.offset)
    def __init__(self, f, a, b, offset=1e-50, *args, **kwargs):
        if f is not None:
            self.wrapped_f = f
        self.orig_a = a
        self.orig_b = b
        self.sign = int(sign(self.wrapped_f(float(a + b) / 2)))
        if b == 0:
            offset = 1e-50
        else:
            offset = abs(b) * finfo(double).eps
        self.offset = offset
        super(PoleInterpolatorN, self).__init__(_wrap_f(self.spec_f),
                                                self.xtinv(self.orig_a), self.xtinv(self.orig_b),
                                                *args, **kwargs)
    def spec_f(self, x):
        return log1p(self.sign * self.wrapped_f(self.xt(x)))
    def interp_at(self, x):
        y = self.sign * expm1(super(PoleInterpolatorN, self).interp_at(self.xtinv(x)))
        return y
    def getNodes(self):
        return self.xt(self.Xs), expm1(self.Ys)
    #def test_accuracy(self, new_Xs, new_Ys):
    #    """Test accuracy by comparing true and interpolated values at
    #    given points."""
    #    errs = abs((super(LogTransformInterpolator, self).interp_at(new_Xs)) - new_Ys)
    #    err = errs.max()
    #    return err
    def diff(self):
        Xs, Ws = chebspace(self.xtinv(self.orig_a), self.xtinv(self.orig_b), self.n, returnWeights=True)
        Ys = concatenate((self.Ys, [self.Ys[-1]]))
        dp = BarycentricInterpolator(Xs, Ys, Ws).diff()
        Xs, Ws = chebspace1(self.orig_a, self.orig_b, self.n, returnWeights=True)
        Ys = exp(super(PoleInterpolatorN, self).interp_at(self.xtinv(Xs))) * dp(self.xtinv(Xs)) /(-(Xs - self.orig_b) + self.offset)*(-1)
        return BarycentricInterpolator(Xs=Xs, Ys=Ys, weights=Ws)
    def roots(self):
        return array([])
    def roots_in_diff(self):
        Xs, Ws = chebspace(self.xtinv(self.orig_a), self.xtinv(self.orig_b), self.n, returnWeights=True)
        Ys = concatenate(([self.Ys[0]], self.Ys))
        dp = BarycentricInterpolator(Xs, Ys, Ws).diff()
        return dp.roots()

# TODO: unused
class ZeroNeighborhoodInterpolator(object):
    """Interpolates f on [0, U].  Splits the interval adaptively to
    get good accuracy around zero."""
    def __init__(self, f, L, U, interp_class=ChebyshevInterpolator, minx=0.5, stop_y=1e-200):
        self.f = f
        self.interp_class = interp_class
        self.a = 0
        self.b = U
        self.interps = []
        Utmp = U
        first_interp = True
        while True:
            Ltmp = Utmp * minx
            print("[", Ltmp, Utmp, "]")
            if first_interp:
                I = interp_class(f, Ltmp, Utmp)
                first_interp = False
            else:
                I = interp_class(f, Ltmp, Utmp, abstol=0)
            self.interps.append(I)
            if I.Ys[0] <= stop_y:
                Ilast = interp_class(f, 0, Ltmp, abstol=0)
                self.interps.append(Ilast)
                break
            Utmp = Ltmp
        Xs = []
        Ys = []
        for I in self.interps:
            Xs += list(I.Xs)
            Ys += list(I.Ys)
        XYs = list(zip(Xs, Ys))
        XYs.sort()
        self.Xs = array([t[0] for t in XYs])
        self.Ys = array([t[1] for t in XYs])

    def interp_at(self, xx):
        if size(xx) == 1:
            xx = array([xx])
        y = zeros_like(xx)
        for j in range(len(xx)):
            x = xx[j]
            for I in self.interps:
                if I.b >= x >= I.a:
                    y[j] = I.interp_at(array([x]))
                    break
        return y
    def __call__(self, x):
        return self.interp_at(x)

class VarTransformInterpolator(ChebyshevInterpolator): # original state
#class VarTransformInterpolator(ChebyshevInterpolator1):
#class VarTransformInterpolator(ZeroNeighborhoodInterpolator):
#class VarTransformInterpolator(ValTransformInterpolator):
    """Interpolator with variable transform."""
    def __init__(self, f, vt=None, par=None):
        self.wrapped_f = f
        if vt is None:
            vt = VarTransformIdentity()
        self.vt = vt
        super(VarTransformInterpolator, self).__init__(_wrap_f(self.spec_f),
                                                       self.vt.var_min,
                                                       self.vt.var_max,
                                                       par=par)
    def spec_f(self, t):
        return self.vt.apply_with_inv_transform(self.wrapped_f, t)
    def transformed_interp_at(self, t):
        """Direct access to transformed function."""
        ret = super(VarTransformInterpolator, self).interp_at(t)
        return ret
    def interp_at(self, x):
        t = self.vt.var_change(x)
        ret = super(VarTransformInterpolator, self).interp_at(t)
        return ret
    def getNodes(self):
        return self.vt.inv_var_change(self.Xs), self.Ys



class ChebyshevInterpolator_PMInf(VarTransformInterpolator):
    def __init__(self, f):
        vt = VarTransformAlgebraic_PMInf()
        super(ChebyshevInterpolator_PMInf, self).__init__(f, vt)
class ChebyshevInterpolator_PInf(VarTransformInterpolator):
    def __init__(self, f, L, exponent=None, U=None):
        if exponent is None:
            exponent = params.interpolation_infinite.exponent
        vt = VarTransformReciprocal_PInf(L, exponent=exponent, U=U)
        super(ChebyshevInterpolator_PInf, self).__init__(f, vt, par=params.interpolation_infinite)
class ChebyshevInterpolator_MInf(VarTransformInterpolator):
    def __init__(self, f, U, exponent=None, L=None):
        if exponent is None:
            exponent = params.interpolation_infinite.exponent
        vt = VarTransformReciprocal_MInf(U, exponent=exponent, L=L)
        super(ChebyshevInterpolator_MInf, self).__init__(f, vt, par=params.interpolation_infinite)

#class ChebyshevInterpolator_PInf(LogTransformInterpolator):
#    def __init__(self, f, L, exponent = None, U = None):
#        super(ChebyshevInterpolator_PInf, self).__init__(f, L, U)
#class ChebyshevInterpolator_MInf(LogTransformInterpolator):
#    def __init__(self, f, U, exponent = None, L = None):
#        if L is not None:
#            L = -L
#        super(ChebyshevInterpolator_MInf, self).__init__(lambda x: f(-x), -U, L)


#PInfInterpolator = ChebyshevInterpolator_PInf
#class PInfInterpolator(LogTransformInterpolator):
#    def __init__(self, f, L, exponent = None, U = None):
#        super(PInfInterpolator, self).__init__(f, L, U)
#        self.U = self.orig_b
class PInfInterpolator(object):
    def __init__(self, f, L, U=None):
        # parameters
        exponent = params.interpolation_infinite.exponent # var transform interpolator's exponent
        max_order = 1e06          # maximum order of difference for Ys in barycentric interpolator
        min_x_barycentric = 1.0 / max_order  # minimum x for which barycentric interpolator is used
        min_nonzero_y = 1e-100    # minimum value of Y considered nonzero

        if f is not None:
            self.f = f
        #Ut = _find_zero(f, L,ymax=1e-14, ymin=1e-15)
        #self.vt = VarTransformReciprocal_PInf(L, U = Ut, exponent = exponent)
        self.vt = VarTransformReciprocal_PInf(L, exponent=exponent)
        self.vb = VarTransformInterpolator(self.f, self.vt, par=params.interpolation_infinite)
        # a barycentric interpolator can fail in two ways:
        # A) x is very close x_j and x-x_j ~= x_j; also x_j <= 1
        # B) if y is << than max(y_j)
        ys = self.vb.Ys[1:] # first Y is always 0
        xs = self.vb.Xs[1:]
        max_y = max(ys)
        if max_y <= min_nonzero_y:
            # don't need an asymptotic interpolator
            self.x_vb_max = None
            self.vl = None
            self.U = Inf
        else:
            x_vb_min = 0 # TODO:  to add somthing like this
            for i, y in enumerate(ys):
                if y > 0 and max_y / y <= max_order:
                    x_vb_min = xs[i]
                    break
            self.x_vb_max = self.vt.inv_var_change(max(min_x_barycentric, x_vb_min))
            #self.x_vb_max = Ut
            # recalculate the interpolator
            self.vt = VarTransformReciprocal_PInf(L, U=self.x_vb_max, exponent=exponent)
            self.vb = VarTransformInterpolator(self.f, self.vt, par=params.interpolation_infinite)
            self.vl = LogTransformInterpolator(self.f, self.x_vb_max, par=params.interpolation_asymp)
            self.U = self.vl.orig_b
        if params.interpolation_asymp.debug_info:
            #print "vb.minmax", L, Ut
            if self.vl is not None:
                print("vl.minmax", self.vl.orig_a, self.vl.orig_b)
            print("self.x_vb_max", self.x_vb_max, end=' ') #self.f(self.x_vb_max)
    def interp_at(self, x):
        if isscalar(x):
            if self.x_vb_max is None or x <= self.x_vb_max:
                y = self.vb(x)
            else:
                y = self.vl(x)
        else:
            if self.x_vb_max is None:
                y = self.vb(x)
            else:
                y = empty_like(x)
                mask = (x <= self.x_vb_max)
                y[mask] = self.vb(x[mask])
                y[~mask] = self.vl(x[~mask])
        return y
    def __call__(self, x):
        return self.interp_at(x)
    def getNodes(self):
        vbXs, vbYs = self.vb.getNodes()
        if self.vl is not None:
            vlXs, vlYs = self.vl.getNodes()
        else:
            vlXs, vlYs = [], []
        Xs = hstack([asarray(vbXs), asarray(vlXs)])
        Ys = hstack([asarray(vbYs), asarray(vlYs)])
        return Xs, Ys
    def __str__(self):
        s = str(self.vb)
        if self.vl is not None:
            s += "\n" + str(self.vl)
        return s

#MInfInterpolator = ChebyshevInterpolator_MInf
class MInfInterpolator(PInfInterpolator):
    def __init__(self, f, U, L=None):
        self.wrapped_f = f
        if L is not None:
            L = -L
        super(MInfInterpolator, self).__init__(_wrap_f(self.spec_f), -U, L)
        self.L = -self.U
        self.U = None
    def spec_f(self, x):
        return self.wrapped_f(-x)
    def interp_at(self, x):
        return super(MInfInterpolator, self).interp_at(-x)
    def getNodes(self):
        Xs, Ys = super(MInfInterpolator, self).getNodes()
        return -Xs, Ys

# the ones below use Boyd's Chebyshev rational functions
#class ChebyshevInterpolator_PInf(VarTransformInterpolator):
#    def __init__(self, f, L):
#        vt = VarTransformAlgebraic_PInf(L)
#        super(ChebyshevInterpolator_PInf, self).__init__(f, vt)
#class ChebyshevInterpolator_MInf(VarTransformInterpolator):
#    def __init__(self, f, U):
#        vt = VarTransformAlgebraic_MInf(U)
#        super(ChebyshevInterpolator_MInf, self).__init__(f, vt)



if __name__ == "__main__":
    from pylab import *
    from pacal import *
#    B= BetaDistr(1,1) * UniformDistr(0,3)
#    B =(UniformDistr(0,3)+UniformDistr(0,1)+UniformDistr(0,1)+UniformDistr(0,1)) * (UniformDistr(0,1)+UniformDistr(0,1)+UniformDistr(0,1))
#    B = BetaDistr(4,4) *  (UniformDistr(-1,1)+UniformDistr(-1,2))
#    B.summary(show_moments=True)
#    print B.get_piecewise_pdf()
    from pacal.segments import PiecewiseFunction
    B = PiecewiseFunction(fun=lambda x:sin(3*x), breakPoints=[-1,0,1])

    B = B.toInterpolated()
    print(B.segments[0].f.__class__)
    #B = B.trimInterpolators(abstol=1e-15)
    print(B.segments[0].f.Ys, B.segments[0].f.__class__)
    D = B.diff()
    D2 = D.diff()
    D3 = D2.diff()
    D4 = D3.diff()
    D5 = D4.diff()
    print(D.segments[0].f.Ys, D.segments[0].f.__class__)
    print(D2.segments[0].f.Ys)
    print(D.roots())
    figure()
    B.plot()
    D.plot()
    D2.plot()
    D3.plot()
    D4.plot()
    D5.plot()
    show()
    0/0
    #Xs = [-1,0,1]
    #Ys = [0,1,0]
    #i = Interpolator(Xs, Ys)
    #X = linspace(-1,1,100)
    #Y = [i.interp_at(x) for x in X]

    #ci = ChebyshevInterpolator(cos, -2, 8)
    #print ci.err, len(ci.Xs)
    #print ci.interp_at(8)
    #X = linspace(ci.a,ci.b,1000)
    #Y = [ci.interp_at(x) for x in X]
    #0/0

    #f = sin
    #f = lambda x: sin(x)**2
    #cii = ChebyshevInterpolator(f, 0, 1)
    #ci = PolyInterpolator(cii, 1e-1, maxdeg = 5)
    #print
    #print ci.err, len(ci.Xs)
    #print cii.err, len(cii.Xs)
    #for x in [1e-11, 1e-20, 1e-50, 1e-100]:
    #    print x, ci.interp_at(x), cii.interp_at(x), f(x)
    #0/0

    # asymptotic behavior for small arguments
    f = sin
    #f = lambda x: sin(x) * sin(x)
    #f = lambda x: x ** 2.5
    U = 1
    #ci = ChebyshevInterpolator(f, 0, U, abstol = 0)
    #ci = ZeroNeighborhoodInterpolator(f, U, ChebyshevInterpolator)
    # #print ci.err, len(ci.Xs)
    #x = 1.5e-50
    #print ci.interp_at(x)
    #print f(x)
    #X = linspace(ci.a,ci.b,1000)
    #Y = [ci.interp_at(x) for x in X]



    #cempty = ChebyshevInterpolator(cos, 1, 1)
    #print cempty(1)
    #0/0

    #ci_pminf = ChebyshevInterpolator_PMInf(lambda x: exp(-abs(x+1)))
    #ci_pminf = ChebyshevInterpolator_PMInf(lambda x: exp(-x*x))
    #print ci_pminf.err, len(ci_pminf.Xs)
    #X = linspace(-10,10,1000)
    #Y = [ci_pminf.interp_at(x) for x in X]
    #Y2 = ci_pminf.interp_at(X)
    #plot(X, Y)
    #plot(X, Y2)
    #show()

    #ci_pinf = ChebyshevInterpolator_PInf(lambda x: exp(-x), 2)
    #print ci_pinf.err, len(ci_pinf.Xs)
    #X = linspace(ci_pinf.L,10,1000)
    #Y = ci_pinf.interp_at(X)
    ##Y2 = [ci_pinf.interp_at(x) for x in X]
    #plot(X, Y)
    ##plot(X, Y2)
    #show()
    #0/0

    #ci_minf = ChebyshevInterpolator_MInf(lambda x: exp(x), -2)
    #print ci_minf.err, len(ci_minf.Xs)
    #X = linspace(-10,ci_minf.U,1000)
    #Y = [ci_minf.interp_at(x) for x in X]

    # Cauchy distr.
    def cauchy(x):
        return 1.0 / (pi * (1 + x * x))
    def normpdf(x, mu=0, sigma=1):
        return 1.0 / sqrt(2 * pi) / sigma * exp(-(x - mu) ** 2 / 2 / sigma ** 2)
    def normpdf_log(x, mu=0, sigma=1):
        return log(normpdf(expm1(x)))
    def prodcauchy(x):
        return 2.0 / (pi * pi * (x * x - 1)) * log(abs(x))
    def prodcauchy_uni(x):
        return 1.0 / (2 * pi) * log1p((1.0 / (x * x)))
    def chisqr(x, kk=1):
        coeffs = [0.0, 2.506628274631001, 2, 2.506628274631, 4, 7.519884823893001, 16, 37.59942411946501, 96, 263.1959688362551, 768, 2368.763719526296, 7680, 26056.40091478925, 92160, 338733.2118922602, 1290240, 5080998.178383904, 20643840, 86376969.03252636, 371589120, 1641162411.618001, 7431782400, 34464410643.97802, 163499212800, 792681444811.4906, 3923981107200, 19817036120287.35, 102023508787200, 535059975247760.3, 2856658246041600, 1.551673928218494e+016, 8.5699747381248e+016, 4.810189177477336e+017, 2.742391916199936e+018, 1.587362428567526e+019, 9.324132515079782e+019, 5.55576849998637e+020, 3.356687705428722e+021, 2.055634344994941e+022, 1.275541328062914e+023, 8.016973945480292e+023, 5.102165312251657e+024, 3.28695931764694e+025, 2.142909431145696e+026, 1.413392506588176e+027, 9.428801497041062e+027, 6.360266279646801e+028, 4.337248688638937e+029, 2.989325151434023e+030, 2.081879370546656e+031, 1.464769324202674e+032, 1.040939685273327e+033, 7.470323553433536e+033, 5.412886363421323e+034, 3.959271483319787e+035, 2.922958636247489e+036, 2.17759931582587e+037, 1.636856836298603e+038, 1.24123161002075e+039, 9.49376965053202e+039, 7.32326649912245e+040, 5.696261790319202e+041, 4.467192564464701e+042, 3.531682309997874e+043, 2.814331315612744e+044, 2.260276678398662e+045, 1.829315355148299e+046, 1.491782607743107e+047, 1.225641287949337e+048, 1.014412173265315e+049, 8.456924886850515e+049, 7.100885212857138e+050, 6.004416669663804e+051, 5.112637353257176e+052, 4.38322416885467e+053, 3.783351641410374e+054, 3.287418126640914e+055, 2.875347247471884e+056, 2.531311957513567e+057, 2.242770853028042e+058, 1.99973644643572e+059, 1.794216682422407e+060, 1.619786521612925e+061, 1.47125767958639e+062, 1.344422812938723e+063, 1.235856450852555e+064, 1.142759390997911e+065, 1.062836547733212e+066, 9.942006701681694e+066, 9.352961620052071e+067, 8.848385964496985e+068, 8.417665458046964e+069, 8.052031227692019e+070, 7.744252221403107e+071, 7.488389041753833e+072, 7.279597088118913e+073, 7.113969589665893e+074, 6.988413204594182e+075, 6.900550501975964e+076, 6.848644940502514e+077]
        if isscalar(x):
            return 0 if x < 0 else 1.0 / coeffs[kk] * x ** (kk / 2.0 - 1.0) * exp(-x / 2.0)
        else:
            y = zeros_like(x)
            y[x >= 0] = 1.0 / coeffs[kk] * x[x >= 0] ** (kk / 2.0 - 1.0) * exp(-x[x >= 0] / 2.0)
        return y
    #ci = ChebyshevInterpolator_PInf(cauchy, 1)
    #print ci.err, len(ci.Xs)

    for pdf in [chisqr, lambda x:-log(x)]:#, cauchy, lambda x: 1.0/(1+x**1.5), prodcauchy]:#, prodcauchy, prodcauchy_uni, chisqr, lambda x: sin(3*x)]:
        print("=======================================")
        x1 = 0.0
        x2 = 0.1
        #x1, x2 = 2.01029912342, 2.82968863313
        #normpdf = cauchy
        #normpdf = prodcauchy
        #normpdf = prodcauchy_uni
        #normpdf = lambda x:chisqr(x,1)
        #ci = ChebyshevInterpolatorNoR2(pdf, x1, x2)
        #cii = ChebyshevInterpolatorNoL2(pdf, x1, x2)
        print("1==============")
        #cii = PInfInterpolator(pdf, 2)
        #mii = MInfInterpolator(pdf, -x1)
        from numpy import log, exp
        cii = LogXChebyshevInterpolator(lambda x: log(pdf(x)), 1e-50, x2)
        print("2==============")
        dii = ChebyshevInterpolator(lambda x: log(pdf(exp(x))), -50, -1)
        #dii = ChebyshevInterpolatorNoL2(pdf, x1, x2)
        #dii = ChebyshevInterpolator(pdf, 2, 10)
        print("3==============")
        #ci = LogXChebyshevInterpolator(pdf, 10, 10000)
        ci = PoleInterpolatorP(pdf, x1, x2)
        #dii = cii
        #ci = cii

        figure()
        plt.title(pdf.__name__)
        #fz = lambda x: 1/x
        #fz = pdf
        #fz = prodcauchy
        #fz = prodcauchy_uni
        #xz = _find_zero(fz, 1); print xz, fz(xz)
        #0/0
        from numpy import log10
        X = logspace(-60, -1, 100000)

        Y1 = ci(X)
        Y2 = exp(cii(X))
        Y3 = dii(log(X))
        Y4 = pdf(X)
        #Y1 = exp(Y1)
        #Y2 = exp(Y2)
        #Y3 = exp(Y3)
        #Y4 = exp(Y4)

        subplot(3, 1, 1)
        Xs, Ys = ci.getNodes()
        plot(X, Y1, 'g', linewidth=3.0)
        plot(Xs, Ys, 'go')

        Xs, Ys = cii.getNodes()
        plot(X, Y2, 'r', linewidth=2.0)
        plot(Xs, Ys, 'rs', markersize=5)

        Xs, Ys = dii.getNodes()
        plot(X, Y3, 'b')
        plot(Xs, Ys, 'b*', markersize=5)
        plot(X, Y4, 'k')


        #plot(cii.Xs, cii.Ys,'ro')
        subplot(3, 1, 2)
        plot(X, abs(Y4 - Y1), 'g', linewidth=3.0)
        plot(X, abs(Y4 - Y2), 'r')
        plot(X, abs(Y4 - Y3), 'b')

        subplot(3, 1, 3)
        plot(X, abs(Y4 - Y1) / Y4, 'g', linewidth=3.0)
        plot(X, abs(Y4 - Y2) / Y4, 'r')
        plot(X, abs(Y4 - Y3) / Y4, 'b')

        from .integration import *
        #print integrate_fejer2_Xn_transformP(cii, x1, 3, N=4) + integrate_fejer2_pinf(cii, 3, x2)
        print(integrate_fejer2_Xn_transformP(pdf, 0, 3, N=4, debug_info=False)[0] + integrate_fejer2_pinf(cii, 3, debug_info=False)[0])
        #print integrate_fejer2_Xn_transformP(dii, x1, x2, N=3)
        #print integrate_fejer2_pinf(cii, x1, x2)
        #print integrate_fejer2_pinf(dii, x1, x2)
        #print "0.9984345977419969"
        #figure()
        #ci.plot_tails()
        #plot(X, abs(cauchy(X)-Y3),'b')
        #print ci.Xs
        #print cii.Xs

        from .integration import *
        #print integrate_fejer2_Xn_transformP(cii, x1, 3, N=4) + integrate_fejer2_pinf(cii, 3, x2)
        #print integrate_fejer2_Xn_transformP(normpdf, 0, 1, N=4)[0] + integrate_fejer2_pinf(cii, 1, x2)[0]
        print("1:", integrate_fejer2_Xn_transformP(normpdf, x1, x2, N=4))
        print("2:", integrate_fejer2_Xn_transformP(dii, x1, x2, N=4))
        #print "3:", integrate_fejer2_Xn_transformN(funneg, -x2, -x1, N=4)
        #print "4:", integrate_fejer2_Xn_transformN(eii, -x2, -x1, N=4)
        #print integrate_fejer2_pinf(cii, x1, x2)
        #print integrate_fejer2(ci, x1, x2)
        #print integrate_fejer2(cii, x1, x2)
        #print integrate_fejer2(dii, x1, x2)
        #print integrate_fejer2(eii,2, 0.0)
        #print "0.9984345977419969"
    #    figure()
    #    X2 = -X
    #    Y5 = funneg(X2)
    #    Xs, Ys = eii.getNodes()
    #    subplot(3,1,1)
    #    plot(X2, eii(X2),'r')
    #    plot(X2, Y5,'b')
    #    plot(Xs, Ys,'rs')

    #    subplot(3,1,2)
    #    plot(X2, abs(Y5-eii(X2)),'b')
    #    subplot(3,1,3)
    #    semilogx(X, abs(Y5-eii(X2))/Y5,'b')
    #print L, M, L/M, normpdf(x), cauchy(x), dii(array([x])), dii.f(x)
    show()
    #print cii.Xs
    #print cii.Ys
