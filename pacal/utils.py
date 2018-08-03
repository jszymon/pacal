#    PaCal - the probabilistic calculator
#    Copyright (C) 2009  Szymon Jaroszewicz, Marcin Korzen
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

import types
from functools import partial
import multiprocessing

from numpy import array, arange, empty, cos, sin, abs
from numpy import pi, isnan, unique, diff
from numpy import hstack, maximum, isfinite
from numpy import isinf, log, exp, logspace, Inf, nan
from numpy import finfo, double, isscalar, asfarray
from pylab import plot, loglog, show, semilogx, sqrt, figure
from pylab import real, ones_like
from numpy import zeros, sort
from numpy.linalg import eigvals
from numpy.fft.fftpack import fft, ifft
from numpy import real, concatenate, linspace, argmin


#from scipy.fftpack.basic import fft
from scipy.optimize import fmin_cg,fmin, fmin_tnc

from . import params

# safe infinity
try:
    from numpy import Inf
except:
    Inf = float('inf')

# function wrappers for avoiding lambdas

# wrap instancemethod .pdf in a partial function call for pickling
# only used for parallel execution
def _call_pdf(obj, x):
    return obj.pdf(x)
def wrap_pdf(pdf):
    if params.general.parallel:
        return partial(_call_pdf, pdf.__self__)
    return pdf


def combine_interpolation_nodes(oldXs, oldYs, newXs, newYs):
    """Combine old and new interpolation nodes in sorted order."""
    XsYs_sorted = sorted(zip(list(oldXs) + list(newXs), list(oldYs) + list(newYs)))
    Xs = array([t[0] for t in XsYs_sorted])
    Ys = array([t[1] for t in XsYs_sorted])
    return Xs, Ys
def combine_interpolation_nodes_fast(oldXs, oldYs, newXs, newYs):
    """Combine old and new interpolation nodes in sorted order."""
    newsize = len(oldXs) + len(newXs)
    combinedXs = empty(newsize)
    combinedYs = empty(newsize)
    combinedXs[::2] = oldXs
    combinedXs[1::2] = newXs
    combinedYs[::2] = oldYs
    combinedYs[1::2] = newYs
    return combinedXs, combinedYs

# Chebyshev related utilities
def cheb_nodes(n, a = -1, b = 1):
    """Chebyshev nodes for given degree n"""
    apb = 0.5 * (a + b)
    if n == 1:
        return array([apb])
    bma = 0.5 * (b - a)
    cs = apb - bma * cos(arange(n) * pi / (n-1))
    # ensure that endpoints are exact
    cs[0] = a
    cs[-1] = b
    return cs

# Chebyshev nodes in logspace
def cheb_nodes_log(n, a = 1, b = 10):
    """Chebyshev nodes in logspace for given degree n"""
    assert(0 < a  < b)
    cs = cos(arange(n) * pi / (n-1))
    cs = exp(cs/2.0*log(b/a))*sqrt(a*b)
    # ensure that endpoints are exact
    cs[0] = a
    cs[-1] = b
    return cs

def chebspace(a, b, n, returnWeights=False):
    """Chebyshev nodes for given degree n"""
    apb = 0.5 * (a + b)
    bma = 0.5 * (b - a)
    cs = apb - bma * cos(arange(n) * pi / (n-1))
    # ensure that endpoints are exact
    cs[0] = a
    cs[-1] = b
    if returnWeights:
        weights = ones_like(cs)
        weights[::2] = -1
        weights[0] /= 2
        weights[-1] /= 2
        return cs, weights
    else:
        return cs

def chebspace1(a, b, n, returnWeights=False):
    """Chebyshev nodes for given degree n"""
    apb = 0.5 * (a + b)
    bma = 0.5 * (b - a)
    cs = apb - bma * cos(arange(1, 2*n, 2) * pi / (2*n))
    if returnWeights:
        weights = ones_like(cs)
        weights = sin(arange(1, 2 * n, 2) * pi / (2 * n))
        weights[1::2] = -1 * weights[1::2]
        return cs, weights
    else:
        return cs

def incremental_cheb_nodes(n, a = -1, b = 1):
    """Extra Chebyshev nodes added by moving from degree m to n=2*m-1"""
    apb = 0.5 * (a + b)
    bma = 0.5 * (b - a)
    return apb - bma * cos(arange(1,n-1,2) * pi / (n-1))

def incremental_cheb_nodes_log(n, a = 1, b = 10):
    """Extra Chebyshev nodes in logspace added by moving from degree m to n=2*m-1"""
    cs = cos(arange(1,n-1,2) * pi / (n-1))
    return exp(cs/2.0*log(b/a))*sqrt(a*b)

def cheb_nodes1(n, a = -1, b = 1):
    """Chebyshev nodes of the first kind for given degree n.

    These are roots of Cheb. polys of the 1st kind."""
    apb = 0.5 * (a + b)
    bma = 0.5 * (b - a)
    return apb - bma * cos(arange(1, 2*n, 2) * pi / (2*n))

def incremental_cheb_nodes1(n, a = -1, b = 1):
    """Extra Chebyshev nodes added by moving from degree m to n=2*m-1"""
    apb = 0.5 * (a + b)
    bma = 0.5 * (b - a)
    ind = arange(0, n)
    return apb - bma * cos((2*ind[((ind % 3) != 1)] + 1)* pi / (2*n))

def chebt2(f):
    """chebyshev transformation, coefficients in expansion using
    Chebyshev polynomials T_n(x), see chebfun for details"""
    n = len(f)
    oncircle = concatenate((f[-1::-1], f[1:-1]))
    fftcoef = real(fft(oncircle))/(2*n-2)
    #print n, len(fftcoef)
    #print fftcoef[n-1:]
    #print fftcoef[n-1:0:-1]
    fftcoef[n-1:0:-1] += fftcoef[n-1:] # z+conj(z)
    return fftcoef[n-1::-1]
    #return c
def ichebt2(c):
    """inverse chebyshev transformation, values of function in Chebyshev
    nodes of the second kind, see chebfun for details"""
    n = len(c)
    oncircle = concatenate(([c[-1]],c[-2:0:-1]/2, c[0:-1]/2));
    v = real(ifft(oncircle));
    f = (n-1)*concatenate(([2*v[0]], v[1:n-1]+v[-1:n-1:-1], [2*v[n-1]] ))
    return f

def chebt1(f):
    #TODO
    """chebyshev transformation, see chebfun"""
    n = len(f)
    oncircle = concatenate((f[-1::-1], f[1:-1]))
    fftcoef = real(fft(oncircle))/(2*n-2)
    return fftcoef[n-1::-1]
def ichebt1(c):
    #TODO
    """inverse chebyshev transformation, see chebfun"""
    n = len(c)
    print("tam===", n)
    oncircle = concatenate((c[-1::-1], c[1:-1]));
    print("v=", oncircle, n)
    v = real(ifft(oncircle));
    print(v)
    print(v[-2:n:-1])
    print("|", v[1:-1])
    f = (n-1)*concatenate(([2*v(1)], v[-2:n:-1]+v[1:-1], 2*v[-1]));
    print("|", f)
    return f

def cheb1companion(c):
    """s_n[f] = sum_{i=0}^n c_i T_i(x)"""
    n = len(c)
    CT = zeros((n-1, n-1))
    CT[0,1] = 1
    i=arange(1,n-2)
    CT[i, i-1] = 0.5
    CT[i, i+1] = 0.5
    i=arange(0,n-1)
    CT[-1, i] = -.5*c[0:-1]/c[-1]
    CT[-1, -2] = CT[-1, -2] + .5
    return CT
def chebroots(c):
    return sort(eigvals(cheb1companion(c)))
def epsunique(tab, eps = params.segments.unique_eps):
    ub = unique(tab[isnan(tab)==False])
    return ub[~isfinite(ub) | hstack((True, (diff(ub)/maximum(1,abs(ub[1:])))>eps))]

def estimateDegreeOfPole(f, x, pos = True, fromTo = None, N = 10, deriv = False, debug_plot = False):
    if fromTo is None:
        if x == 0:
            fromTo = (-1,-10)
        else:
            # testing around nonzero singularities is less accurate
            fromTo = (-1,-7)
    ex = logspace(fromTo[0], fromTo[1], N)
    if pos:
        lx = x + ex
    else:
        lx = x - ex
    y = abs(f(lx))
    #if deriv:
    #    y -= min(y[isfinite(y)])
    yi = log(y)
    xi = log(abs(ex))
    ind = isfinite(yi)
    xi = xi[ind]
    yi = yi[ind]
    ri = yi[0:-1] - yi[1:]
    di = abs(xi[1:]-xi[0:-1])
    if debug_plot:
        print(xi,yi, f(xi))
        loglog(xi,yi)
    if len(yi) > 1:
        return ri[-1]/di[-1]
    else:
        return 0

def estimateAtInfExponent(f, x, pos = True, fromTo = None, N = 10, deriv = False, debug_plot = False):
    if fromTo is None:
        fromTo = (1,10)
    ex = logspace(fromTo[0], fromTo[1], N)
    if pos:
        lx = ex
    else:
        lx = -ex
    y = abs(f(lx))
    #if deriv:
    #    y -= min(y[isfinite(y)])
    yi = log(y)
    xi = log(abs(ex))
    ind = isfinite(yi)
    xi = xi[ind]
    yi = yi[ind]
    ri = yi[0:-1] - yi[1:]
    di = abs(xi[1:]-xi[0:-1])
    if debug_plot:
        print(xi,yi, f(xi))
        loglog(xi,yi)
    if len(yi) > 1:
        return ri[-1]/di[-1]
    else:
        return 0

def testPole(f, x, pos = True, pole_eps = None, deriv = None, debug_info = None, **kwargs):
    if pole_eps is None:
        pole_eps = params.pole_detection.max_pole_exponent
    if deriv is None:
        deriv = params.pole_detection.derivative
    if debug_info is None:
        debug_info = params.segments.debug_info
    deg = estimateDegreeOfPole(f, x, pos, deriv = deriv, **kwargs)
    if deriv:
        if (abs(deg) >= abs(pole_eps) and deg <= 1 - abs(pole_eps)) or (deg >= 1 + abs(pole_eps) and deg <= 2 - abs(pole_eps)) or deg>2:
        #if (deg >= abs(pole_eps) and deg <= 1 - abs(pole_eps)) or (deg >= 1 + abs(pole_eps) and deg <= 2 - abs(pole_eps))or (deg >= 2 + abs(pole_eps) and deg <= 3 - abs(pole_eps)):
            pole = True
        else:
            pole = False
    else:
        if deg >= pole_eps:
            pole = False
        else:
            pole = True
    if debug_info:
        print("x={0}, deg={1}, pole={2} check_deriv={3}".format(x, deg, pole, deriv))
    return pole

class convergence_monitor(object):
    """Monitor numerical convergence."""
    def __init__(self, par = None):
        # convergence parameters
        if par is None:
            par = params.convergence
        self.abstol = par.abstol
        self.reltol = par.reltol
        self.min_quit_iter = par.min_quit_iter # the earliest iteration to quit early
        self.min_quit_no_improvement = par.min_quit_no_improvement # quit early if no improvement for this # of steps
        self.min_improvement_ratio = par.min_improvement_ratio # by what factor the error needs to improve
        self.min_quit_iter = max(2, self.min_quit_iter)
        self.force_nonzero = par.force_nonzero

        self.ae_list = []
        self.y_list = []
        self.e_list = []

        self.converged = False
        self.n_no_improvement = 0 # for how many steps there was no improvement
        self.last_good = 0 # last entry for which error decreased
    def add(self, abserr, yest, extra_data = None):
        self.ae_list.append(abserr)
        self.y_list.append(yest)
        self.e_list.append(extra_data)
    def test_convergence(self):
        yest = abs(self.y_list[-1])
        ae = self.ae_list[-1]
        step = len(self.ae_list)
        tol = max(self.abstol, yest * self.reltol)
        if ae <= tol and (not self.force_nonzero or yest > 0):
            self.converged = True
            return True, "converged"
        if len(self.ae_list) > 0:
            if ae < self.min_improvement_ratio * self.ae_list[self.last_good]:
                self.last_good = len(self.ae_list) - 1
                self.n_no_improvement = 0
            else:
                self.n_no_improvement += 1
        if step >= self.min_quit_iter:
            if self.n_no_improvement >= self.min_quit_no_improvement:
                return True, "did not converge"
        return False, "continue"
    def get_best_result(self, err_decr = 0.75):
        """Return currently best result.  A result is considered only
        if its error is err_decr times better than previous best."""
        if self.converged:
            return self.y_list[-1], self.ae_list[-1], self.e_list[-1]
        best_y = self.y_list[0]
        best_ae = self.ae_list[0]
        if best_y == 0:
            best_re = finfo(double).max
        else:
            best_re = best_ae / abs(best_y)
        best_e = self.e_list[0]
        for i in range(1, len(self.ae_list)):
            y = self.y_list[i]
            ae = self.ae_list[i]
            if y == 0:
                re = finfo(double).max
            else:
                re = ae / abs(y)
            if re < err_decr * best_re:
                best_y = y
                best_ae = ae
                best_re = re
                best_e = self.e_list[i]
        return best_y, best_ae, best_e
def stepfun(x, shift = 0.0):
    if isscalar(x):
        if x < shift:
            return 0.0
        else:
            return 1.0
    else:
        mask = (x >= 0.0)
        y = zeros_like(asfarray(x))
        y[mask] = 1.0
        return y

# Root finding

try:
    from scipy.optimize import ridder, brentq
    have_scipy_opt = True
except:
    have_scipy_opt = False
#have_scipy_opt = False
def findinv(fun, a = 0.0, b = 1.0, c = 0.5, **kwargs):
    """find solution of equation f(x)=c, on interval [a, b]"""
    if have_scipy_opt:
        # fix too low relative tolerance for brentq
        if "rtol" in kwargs and kwargs["rtol"] < finfo(float).eps * 4:
            kwargs["rtol"] = finfo(float).eps * 4
        return brentq(lambda x : fun(x) - c, a, b, **kwargs)
    else:
        return bisect(lambda x : fun(x) - c, a, b, **kwargs)
def findinv_minf(fun, b = 0.0, c=0.5, **kwargs):
    """Find inverse fun(x)==c below b, possibly at -oo."""
    a = b - max(1, 0.1*abs(b))
    fa = fun(a)
    fb = fun(b)
    if not fun(b) >= c:  # assume fun decreasing
        return nan
    while isfinite(a) and fa > c:
        d = (b - a) * 1.2
        fb = fa
        a, b = a-d, a
        fa = fun(a)
    if isfinite(a):
        x = findinv(fun, a, b, c, **kwargs)
    else:
        x = a
    return x
def findinv_pinf(fun, a = 0.0, c=0.5, increasing=True, **kwargs):
    """Find inverse fun(x)==c below b, possibly at -oo."""
    if increasing:
        s = 1
    else:
        s = -1
    b = a + max(1, 0.1*abs(a))
    fa = fun(a)
    fb = fun(b)
    if not s*(fun(a) - c) <= 0:  # assume fun decreasing
        return nan
    while isfinite(b) and fb < c:
        d = (b - a) * 1.2
        fa = fb
        a, b = b, b+d
        fb = fun(b)
    if isfinite(b):
        x = findinv(fun, a, b, c, **kwargs)
    else:
        x = b
    return x
# copied from scipy
def bisect(f, xa, xb, xtol = 10*finfo(double).eps, rtol = 2*finfo(double).eps, maxiter = 1000, args = ()):
    tol = min(xtol, rtol*(abs(xa) + abs(xb))) # fix for long intervals
    fa = f(xa, *args)
    fb = f(xb, *args)
    if fa*fb > 0: raise RuntimeError("Interval does not contain zero")
    if fa == 0: return xa
    if fb == 0: return xb

    dm = xb - xa
    for i in range(maxiter):
        dm /= 2
        xm = xa + dm
        fm = f(xm, *args)
        if fm*fa >= 0:
            xa = xm
        if fm == 0 or abs(dm) < tol:
            return xm
    print("WARNING: zero fidning did not converge")
    return xm

def estimateTailExponent(f, fromTo = None, N =300, deriv = False, debug_plot = False, pos = True):
    if fromTo is None:
        fromTo = (1,100)
    ex = logspace(fromTo[0], fromTo[1], N)
    if pos:
        lx = ex
        xi = log(ex)
    else:
        lx = -ex
        xi = -log(ex)
    y = abs(f(lx))
    ind = (y > 0)
    xi = xi[ind]
    yi = log(y[ind])
    ri = yi[1:] - yi[0:-1]
    di = abs(xi[1:]-xi[0:-1])
    if debug_plot:
        print(ri, di)
        plot(xi,yi)
    if len(yi) > 1:
        ex = ri[-1]/di[-1]
        if ex>50:
            return Inf
        else:
            return ex
    else:
        return 0

def maxprob(pdf, x0, lub=None):
    def fun(x):
        #print x, lub
        if lub is not None:
            for i in range(len(x)):
                if x[i]<lub[i][0]:
                    x[i]=lub[i][0]
                if x[i]>lub[i][1]:
                    x[i]=lub[i][1]

        f = -pdf(*[x[i] for i in range(len(x))])
        #print x, f
        return f
    #return fmin_tnc(fun, x0,bounds=lub)
    return fmin_cg(fun, x0, gtol=1e-14)
    #return fmin(fun, x0)
def fmin2(fc, L, U, **kwargs):
    xx = linspace(L,U,20)
    y = [fc(x) for x in xx]
    ind = argmin(y)
    #xopt = fmin_cg(fc, xx[ind], maxiter=20, disp=0)
    xopt = fmin(fc, xx[ind], maxiter=20, disp=0)
    if xopt>U:
        return U
    if xopt<L:
        return L
    return xopt


try:
    from math import lgamma
except:
    from .gamma import lgamma

def binomial_coeff(n, k):
    if k > n - k: # take advantage of symmetry
        k = n - k
    c = 1
    for i in range(k):
        c = c * (n - i)
        c = c / (i + 1)
    return c

def multinomial_coeff(n, ki=[]):
    assert sum(ki)==n, "incorrect values n, k ({0}, {1})".format(n, ki)
    c = 1
    j=0
    for k in ki:
        for i in range(k):
            c = c * (n - j)
            c = c / (i + 1)
            j += 1
    return c

def taylor_coeff(fun, N):
    """From L. Trefethen, Ten digits algorithms """
    zz = exp(2j*pi*(array(list(range(N))))/N)
    c = fft(fun(zz))/N
    return real(c)

_debug_fig = None
_debug_cancelled = False
def debug_plot(a, b, nodes, fs, coeffs):
    global _debug_fig, _debug_cancelled
    if _debug_cancelled:
        return
    if 'show' not in locals():
        from pylab import axes, subplot, subplots_adjust, figure, draw, plot, axvline, xlim, title, waitforbuttonpress, gcf
        from matplotlib.widgets import Button
    if _debug_fig is None:
        #curfig = gcf()
        #print dir(curfig)
        _debug_fig = figure()
        ax = _debug_fig.add_subplot(111)
        #subplots_adjust(bottom=0.15)
        butax = axes([0.8, 0.015, 0.1, 0.04])
        button = Button(butax, 'Debug', hovercolor='0.975')
        def debug(event):
            import pdb; pdb.set_trace()
        button.on_clicked(debug)
        _debug_fig.sca(ax)
        draw()
        #figure(curfig)
    _debug_fig.gca().clear()
    plot(nodes, fs, linewidth=5, figure = _debug_fig)
    axvline(a, color="r", figure = _debug_fig)
    axvline(b, color="r", figure = _debug_fig)
    d = 0.05 * (b-a)
    _debug_fig.gca().set_xlim(a-d, b+d)
    title("press key in figure for next debugplot or close window to continue")
    try:
        while not _debug_cancelled and not _debug_fig.waitforbuttonpress(-1):
            pass
    except:
        _debug_cancelled = True

def ordinal_ending(n):
    if n == 1:
        return "st"
    if n == 2:
        return "nd"
    return "th"


def is_instance_method(obj):
    """Checks if an object is a bound method on an instance."""
    if not isinstance(obj, types.MethodType):
        return False # Not a method
    if obj.__self__ is None:
        return False # Method is not bound
    if issubclass(obj.__self__.__class__, type) or obj.__self__.__class__ is type:
        return False # Method is a classmethod
    return True

def list_map(*args):
    """map returning a list in line with multiprocessing Pool.map"""
    return list(map(*args))
def par_map(f, x):
    if len(x) < 100:
        return list_map(f, x)
    return params.general.process_pool.map(f, x)
def init_pacal_thread():
    import multiprocessing, os
    worker_name = "Pacal__worker__" + str(os.getpid())
    multiprocessing.current_process().name = worker_name
def get_parmap():
    if params.general.parallel:
        p = multiprocessing.current_process()
        if p.name.startswith("Pacal__worker__"):
            pmap = list_map
        else:
            if params.general.process_pool is None:
                params.general.process_pool = multiprocessing.Pool(params.general.nprocs,
                                                                   initializer=init_pacal_thread)
            pmap = params.general.process_pool.map
    else:
        pmap = list_map
    return pmap


if __name__ == "__main__":
    from pacal import *
    import time
    import numpy.polynomial.chebyshev as ch
    c = array([1, 2, 3, 1])
    CT =cheb1companion(array([1, 2, 3, 1]))
    print(CT)
    print(chebroots(c))
    print(ch.chebroots(c))
    print(chebroots(c) - ch.chebroots(c))
    0/0
    #print taylor_coeff(lambda x:exp(x), 30)
    N = NormalDistr()
    fun  = N.mgf()
    #fun.plot()
    t0  = time.time()
    t_i =  taylor_coeff(fun, 100)
    print(time.time()-t0)
    sil=1

    t0  = time.time()
    for i in range(50):
        if i==0:
            sil=1
        else:
            sil *= i
        mi = N.moment(i, 0.0)
        print(i, repr(mi),  repr(t_i[i])*sil*2,  repr(mi/sil/2), repr(t_i[i]), repr(mi/sil/2-t_i[i]));
    print(time.time()-t0)


    print(N.summary())
    show()
    0/0
    print(binomial_coeff(10, 7))
    print(multinomial_coeff(10, [3, 3, 4]))
    print(multinomial_coeff(13, [7, 2, 4]))
    print(multinomial_coeff(21, [9, 8, 4]))
    0/0

    from .standard_distr import *
    from pylab import *

    print(estimateTailExponent(LevyDistr(), pos = True))
    L = LevyDistr()
    L.summary()

    A= UniformDistr() / UniformDistr()
 # ChiSquareDistr(1) / ChiSquareDistr(1.1)
    A.summary()
    A.plot()
    S =A
    figure()
    for i in linspace(1,10,10):
        S_1 = S * 2
        S = S + S
        subplot(211)
        (S/(2**(i))).plot(xmin=0,xmax=50)
        print(i, end=' ')
        (S/(2**(i))).summary()
        subplot(212)
        r = S.get_piecewise_pdf() - S_1.get_piecewise_pdf()
        r.plot(xmin=0,xmax=50)
    show()
    0/0
    #m = 3
    #n = 2*m-1
    #print cheb_nodes(m)
    #print incremental_cheb_nodes(n)
    #print cheb_nodes(n)
    #print combine_interpolation_nodes(cheb_nodes(m),
    #                                  arange(m),
    #                                  incremental_cheb_nodes(n),
    #                                  arange(m-1))[0]
    #print combine_interpolation_nodes_fast(cheb_nodes(m),
    #                                       arange(m),
    #                                       incremental_cheb_nodes(n),
    #                                       arange(m-1))[0]

    #from pylab import plot, show, axvline, figure
    #for i in xrange(2,5):
    #    print i, cheb_nodes1(i)
    #    plot(cheb_nodes1(i), [i]*i, "o")
    #    for x in cheb_nodes1(i):
    #        axvline(x)
     #segf1 = Segment(0.0, 1.0, lambda x:(n+1)/(n) * x ** (1/n))
        #segf2 = Segment(0.0, 2.0, lambda x:pi/2 * sqrt(1 - (x-1) ** 2))
        #segf1 = Segment(0.0, 1.0, lambda x: exp(-1/x))
    #segf2 = Segment(0.0, 0.5, lambda x:-1/log(x))
    #figure()
    #print estimateDegreeOfZero(lambda x: x**0.5, 0)
    #n=7.0
    #estimateDegreeOfZero(lambda x:(n+1)/(n) * x ** (1/n), 0)
    #estimateDegreeOfZero(lambda x:pi/2 * sqrt(1 - (x-1) ** 2), 0)
    #print estimateDegreeOfZero(lambda x: exp(-1/x), 0)
    # estimateDegreeOfZero(lambda x: 1/(x+x**4), Inf)
    # estimateDegreeOfZero(lambda x: exp(-x), Inf)
    #print findinv(lambda x: 1/(1+exp(-x)), a=-1e300, b=1e300, c=0.5, rtol =1e-16, maxiter = 10000)

    from numpy import ones_like, zeros_like
    def _pole_test(f, x, pos = True, deriv = False):
        return str(testPole(f, x, pos, deriv = deriv)) + "   " + str(estimateDegreeOfPole(f, x, pos)) + "   " + str(estimateDegreeOfPole(f, x, pos, deriv = True))
    print("0,", _pole_test(lambda x: ones_like(x), 0))
    print("0',", _pole_test(lambda x: ones_like(x), 0, deriv = True))
    print("1,", _pole_test(lambda x: zeros_like(x), 0))
    print("x,", _pole_test(lambda x: x, 0))
    print("x',", _pole_test(lambda x: x, 0, deriv = True))
    print("x**1.5,", _pole_test(lambda x: x**1.5, 0))
    print("x**0.5,", _pole_test(lambda x: x**0.5, 0))
    print("-log(x),", _pole_test(lambda x: -log(x), 0))
    print("-log(sqrt(x)),", _pole_test(lambda x: -log(sqrt(x)), 0))
    print("-log(-x),", _pole_test(lambda x: -log(-x), 0, pos = False))
    print("1+x**0.5,", _pole_test(lambda x: 1+x**0.5, 0))
    print("(1+x**0.5)',", _pole_test(lambda x: 1+x**0.5, 0, deriv = True))
    print("x*log(x),", _pole_test(lambda x: x*log(x), 0))
    print(_pole_test(lambda x: 1+(1*x+7)*x**-2.5, 0))
    print(testPole(lambda x: 1.0/abs(2*x-1), 0.5, pos= False))
    print(testPole(lambda x: 9.0*abs(2*x-1), 0.5, pos= True))
