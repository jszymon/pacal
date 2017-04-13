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

from numpy.fft import ifft
from numpy import array, zeros, ones, hstack, arange, zeros_like, asfarray
from numpy import dot, isscalar
from numpy import finfo, double, isinf
from numpy import ceil,log10, logspace

from utils import cheb_nodes, incremental_cheb_nodes
from utils import combine_interpolation_nodes_fast
from utils import convergence_monitor
from utils import debug_plot

from vartransforms import VarTransformAlgebraic_PMInf
from vartransforms import VarTransformAlgebraic_PInf
from vartransforms import VarTransformAlgebraic_MInf
from vartransforms import VarTransformReciprocal_PMInf
from vartransforms import VarTransformReciprocal_PInf
from vartransforms import VarTransformReciprocal_MInf

from pacal.interpolation import ChebyshevInterpolator1
import params

# cache for Clenshaw quadrature coefficients
_clenshaw_cache = {}
def clenshaw_coefficients(n):
    """Return Clenshaw quadrature coefficients for given n.

    Computed values are cached for later reuse.  Based on Waldvogel's
    paper."""
    if n in _clenshaw_cache:
        coeffs = _clenshaw_cache[n]
    else:
        n1 = n - 1
        N = arange(1, n1, 2)
        l = len(N)
        m = n1 - l
        v0 = -hstack([2.0 / (N*(N-2)), [1.0/N[-1]], zeros(m)])
        v2 = v0[:-1] + v0[-1:0:-1]

        g0 = -ones(n1)
        g0[l] += n1
        g0[m] += n1
        g = g0/(n1**2 - 1 + n1%2)
        coeffs = ifft(v2 + g).real
        coeffs = hstack([coeffs, [coeffs[0]]])
        _clenshaw_cache[n] = coeffs
    return coeffs

# cache for Fejer's 2nd rule quadrature coefficients
_fejer2_cache = {}
def fejer2_coefficients(n):
    """Return Fejer's 2nd rule quadrature coefficients for given n.

    Computed values are cached for later reuse.  Based on Waldvogel's
    paper.  First and last coefficients are zero so are removed."""
    if n in _fejer2_cache:
        coeffs = _fejer2_cache[n]
    else:
        n1 = n - 1
        N = arange(1, n1, 2)
        l = len(N)
        m = n1 - l
        v0 = -hstack([2.0 / (N*(N-2)), [1.0/N[-1]], zeros(m)])
        v2 = v0[:-1] + v0[-1:0:-1]
        coeffs = ifft(v2).real
        coeffs = coeffs[1:]
        _fejer2_cache[n] = coeffs
    return coeffs


def integrate_clenshaw(f, a, b, maxn = 2**16, tol = 10e-16,
                       debug_info = True, debug_plot = True):
    n = 3
    prevI = None
    nodes = None
    cm = convergence_monitor()
    while n <= maxn:
        coeffs = clenshaw_coefficients(n)
        if nodes is None:
            nodes = cheb_nodes(n, a, b)
            fs = f(nodes)
        else:
            new_nodes = incremental_cheb_nodes(n, a, b)
            new_fs = f(new_nodes)
            nodes, fs = combine_interpolation_nodes_fast(nodes, fs,
                                                         new_nodes, new_fs)

        I = dot(fs, coeffs) * 0.5 * (b - a)
        if prevI is not None:
            err = abs(I - prevI)
            cm.add(err, I)
            if cm.test_convergence()[0]:
                break
        prevI = I
        n = 2 * n - 1
    if debug_info:
        print "===="
    if debug_plot and n >= maxn: # problems with integration
        debug_plot(a, b, nodes, fs, coeffs)
    return I, err

def integrate_fejer2(f, a, b, par = None, maxn = 2**10, tol = finfo(double).eps,
                     debug_info = False, debug_plot = False):
    if par is not None:
        maxn = par.maxn
        debug_plot = par.debug_plot
        debug_info = par.debug_info
        cm = convergence_monitor(par = par.convergence)
    else:
        cm = convergence_monitor()
    n = 65
    prevI = None
    nodes = None
    while n <= maxn:
        coeffs = fejer2_coefficients(n)
        if nodes is None:
            nodes = cheb_nodes(n, a, b)[1:-1]
            fs = f(nodes)
        else:
            new_nodes = incremental_cheb_nodes(n, a, b)
            new_fs = f(new_nodes)
            # roles of new and old nodes are reversed in the call below
            nodes, fs = combine_interpolation_nodes_fast(new_nodes, new_fs,
                                                         nodes, fs)

        I = dot(fs, coeffs) * (b - a) / 2
        if prevI is not None:
            err = abs(I - prevI)
            if debug_info:
                print repr(I), err, n, I + err == I, err <= abs(I) * tol, min(nodes), max(nodes), min(fs), max(fs)
            cm.add(err, I)
            if cm.test_convergence()[0]:
                break
        prevI = I
        n = 2 * n - 1
    if debug_info:
        print "===="
    if debug_plot and n >= maxn: # problems with integration
        debug_plot(a, b, nodes, fs, coeffs)
    # return currently best result
    I, err, _extra = cm.get_best_result()
    return I, err



def _integrate_with_vartransform(f, vt, quad_routine, *args, **kwargs):
    """Clenshaw integration with variable transform."""
    def __f_int(t):
        y = vt.apply_with_inv_transform(f, t, mul_by_deriv = True)
        return y
    return quad_routine(__f_int, vt.var_min, vt.var_max, *args, **kwargs)

def integrate_clenshaw_pminf(f):
    """Clenshaw integration from -oo to +oo."""
    vt = VarTransformAlgebraic_PMInf()
    return _integrate_with_vartransform(f, vt, integrate_clenshaw)
def integrate_clenshaw_pinf(f, a):
    """Clenshaw integration from a to +oo."""
    vt = VarTransformAlgebraic_PInf(a)
    return _integrate_with_vartransform(f, vt, integrate_clenshaw)
def integrate_clenshaw_minf(f, b):
    """Clenshaw integration from -oo to b."""
    vt = VarTransformAlgebraic_MInf(b)
    return _integrate_with_vartransform(f, vt, integrate_clenshaw)



def integrate_fejer2_pminf(f, *args, **kwargs):
    """Fejer2 integration from -oo to +oo."""
    vt = VarTransformReciprocal_PMInf()
    return _integrate_with_vartransform(f, vt, integrate_fejer2, *args, **kwargs)
def integrate_fejer2_pinf(f, a, b = None, exponent = None, *args, **kwargs):
    """Fejer2 integration from a to +oo."""
    if exponent is None:
        exponent = params.integration_infinite.exponent
    vt = VarTransformReciprocal_PInf(a, U = b, exponent = exponent)
    return _integrate_with_vartransform(f, vt, integrate_fejer2, *args, **kwargs)
def integrate_fejer2_minf(f, b, a = None, exponent = None, *args, **kwargs):
    """Fejer2 integration from -oo to b."""
    if exponent is None:
        exponent = params.integration_infinite.exponent
    vt = VarTransformReciprocal_MInf(b, L = a, exponent = exponent)
    return _integrate_with_vartransform(f, vt, integrate_fejer2, *args, **kwargs)


def integrate_fejer2_Xn_transform(f, a, b, N = 2.0, *args, **kwargs):
    return integrate_fejer2(lambda t: N * (b-a) * f(t**N*(b-a)+a) * t**(N-1), 0.0, 1.0, *args, **kwargs)
def integrate_fejer2_Xn_transformP(f, a, b, N = None, *args, **kwargs):
    if N is None:
        N = params.integration_pole.exponent
    a = a + abs(a) * finfo(double).eps # don't touch the edge
    return integrate_fejer2(lambda t: N * (b-a) * f(t**N*(b-a)+a) * t**(N-1), 0.0, 1.0, *args, **kwargs)
def integrate_fejer2_Xn_transformN(f, a, b, N = None, *args, **kwargs):
    if N is None:
        N = params.integration_pole.exponent
    b = b - abs(b) * finfo(double).eps # don't touch the edge
    return integrate_fejer2(lambda t: N * (b-a) * f(b-t**N*(b-a)) * t**(N-1), 0.0, 1.0, *args, **kwargs)
def integrate_wide_interval(f, a, b, *args, **kwargs):
    wide_cond = params.integration.wide_condition
    if isinf(b):
        b=1e100
    if isinf(a):
        a=-1e100
    if a!=0 and b!=0:
        if (b/a)>wide_cond and 0 < a < b: # positive innterwal
            exp_wide = b/a
            number_of_intervals = ceil(log10(exp_wide)/log10(wide_cond))
            nodes = logspace(log10(a), log10(b), number_of_intervals + 1)
            nodes[0] = a
            nodes[-1] = b
            I,E=0,0
            for i in range(int(number_of_intervals)):
                integ,err = integrate_fejer2(f, nodes[i], nodes[i+1], *args, **kwargs)
                I+=integ
                E+=err
            return I, E
        elif (b/a)<1/wide_cond and a < b < 0: # negative interval
            exp_wide = a/b
            number_of_intervals = ceil(log10(exp_wide)/log10(wide_cond))
            nodes = -logspace(log10(abs(a)), log10(abs(b)), number_of_intervals + 1)
            nodes[0] = a
            nodes[-1] = b
            I,E=0,0
            for i in range(int(number_of_intervals)):
                integ,err = integrate_fejer2(f, nodes[i], nodes[i+1], *args, **kwargs)
                I+=integ
                E+=err
            return I, E
            #print "wide interval neg", a, b, number_of_intervals, nodes, I, E
        else:
            return integrate_fejer2(f, a, b, *args, **kwargs )
    else:
        return integrate_fejer2(f, a, b, *args, **kwargs )

def integrate_wide_interval2(f, a, b, *args, **kwargs):
    m = (a + b) / 2
    ya = f(array([a]))
    ym = f(array([m]))
    yb = f(array([b]))
    if ym < min(ya, yb):
        i1, e1 = integrate_fejer2_pinf(f, a, b = m)
        i2, e2 = integrate_fejer2_minf(f, b, a = m)
    elif ym > max(ya, yb):
        i1, e1 = integrate_fejer2_minf(f, m, a = a)
        i2, e2 = integrate_fejer2_pinf(f, m, b = b)
    elif yb > max(ya, ym):
        i1, e1 = integrate_fejer2_minf(f, b, a = a)
        i2, e2 = 0, 0
    elif ya > max(ym, yb):
        i1, e1 = integrate_fejer2_pinf(f, a, b = b)
        i2, e2 = 0, 0
    else:
        i1, e1 = integrate_fejer2(f, a, b)
        i2, e2 = 0, 0
    i = i1 + i2
    e = e1 + e2
    return i, e

def integrate_iter(f, a1, b1, a2, b2):
    def fun1(y):
        if isscalar(y):
            z = integrate_fejer2(lambda x : f(x, y), a1, b1)
        else:
            z = zeros_like(y)
            for i in range(len(y)):
                z[i], err = integrate_fejer2(lambda x : f(x, y[i]), a1, b1)
        return z
    cheb  = ChebyshevInterpolator1(fun1, a2,b2)
    return integrate_fejer2(cheb, a2, b2)

def integrate_iter2(f, a1, b1, a2, b2):
    def fun1(x):
        if isscalar(x):
            z = integrate_fejer2(lambda y : f(x, y), a2, b2)
        else:
            z = zeros_like(x)
            for i in range(len(x)):
                print ";;;", i, len(x)
                z[i], err = integrate_fejer2(lambda y : f(x[i], y), a2, b2)
        return z
    cheb  = ChebyshevInterpolator1(fun1, a1,b1)
    return integrate_fejer2(cheb, a1, b1)

def integrate_with_pminf_guess(f, a, b, *args, **kwargs):
    """Automatically guess if pinf or minf transformation should be used."""
    ya = f(array([a]))
    yb = f(array([b]))
    if ya > yb:
        i, e = integrate_fejer2_pinf(f, a, b = b, *args, **kwargs)
    else:
        i, e = integrate_fejer2_minf(f, b, a = a, *args, **kwargs)
#    if ya > yb:
#        i, e = integrate_wide_interval(f, a, b, *args, **kwargs)
#    else:
#        i, e = integrate_wide_interval(f, a, b, *args, **kwargs)
    return i, e

if __name__ == "__main__":
    def test_clenshaw(f, a, b, exact):
        """Test clenshaw integration"""
        I, err = integrate_clenshaw(f, a, b)
        print I - exact, err
    def test_fejer2(f, a, b, exact):
        """Test fejer 2nd rule integration"""
        I, err = integrate_fejer2(f, a, b)
        print I - exact, err
    from numpy import sin, cos, pi, exp, sqrt
    test_clenshaw(cos, 0, pi/2, 1)
    test_clenshaw(cos, 0, 2, sin(2))
    test_fejer2(cos, 0, pi/2, 1)
    test_fejer2(cos, 0, 2, sin(2))
    test_fejer2(cos, 0.1, 0.1, 0)
    test_fejer2(cos, 0.1, 0.3/3, 0)
    test_fejer2(cos, 2, 0, -sin(2))

    def cauchy(x):
        return 1.0/(pi*(1.0+x*x))
    def normpdf(x):
        return 1.0/sqrt(2*pi)*exp(-x*x/2)
    I, err = integrate_clenshaw(cauchy, 0, 100)
    print I, err
    # those don't work well with clenshaw because of endpoints:
    #I, err = integrate_clenshaw_pminf(cauchy)
    #print I, err
    #I, err = integrate_clenshaw_pinf(cauchy, 0)
    #print I, err
    #I, err = integrate_clenshaw_minf(cauchy, 0)
    #print I, err
    I, err = integrate_clenshaw_pminf(normpdf)
    print I, err
    I, err = integrate_clenshaw_pinf(normpdf, 0)
    print I, err
    I, err = integrate_clenshaw_minf(normpdf, 0)
    print I, err

    ## Current variable transform won't work for Cauchy on infinite
    ## integral.  Will work fine on half infinite integrals though
    #I, err = integrate_fejer2_pminf(cauchy)
    #print I, err
    I, err = integrate_fejer2_pinf(cauchy, 0)
    print I, err
    I, err = integrate_fejer2_minf(cauchy, 0)
    print I, err
    I, err = integrate_fejer2_pminf(normpdf)
    print I, err
    I, err = integrate_fejer2_pinf(normpdf, 0)
    print I, err
    I, err = integrate_fejer2_minf(normpdf, 0)
    print I, err
    from testPiecewise import *
    I, err = integrate_fejer2_Xn_transformP(lambda x: -log(x), 0.0, 0.1, N=4.0)
    print "log:", I, err
    I, err = integrate_fejer2_Xn_transformP(lambda x: -log(x)/x**0.5, 0.0, 1.0, N=4.0)
    print I, err
    I, err = integrate_fejer2_Xn_transformP(lambda x: chisqr(x), 0.0, 10, N=4.0)
    print "chisqr1=",I, err
    def prodcauchy(x):
        return 1.0/(pi*pi*(x*x-1))*log(x*x)
    from numpy import log1p
    def prodcauchy_uni(x):
        return 1.0/(2*pi)*log1p((1.0/(x*x)))
    I, err = integrate_fejer2_Xn_transform(lambda x: prodcauchy_uni(x), 0.0, 2, N=3.0)
    print I, err
    I, err = integrate_fejer2_Xn_transformP(lambda x: prodcauchy_uni(x), 0.0, 2.0, N=3.0)
    print I, err
    I, err = integrate_fejer2_Xn_transformN(lambda x: prodcauchy_uni(x), -2.0, 0.0, N=3.0)
    print I, err
    #I1, err = integrate_fejer2(lambda x: 1/x, 1e-16, 1e-8)
    #print I1, err
    #I, err = integrate_fejer2(lambda x: 1/x, 1e-8, 1)
    #print I+I1
    #I, err = integrate_wide_interval(lambda x: 1/x, 1e-36,1)
    #print I + log(1e-36)
    def ff(x):
        #print x
        return 0.25/x**0.5/(1.0-x)**0.5
    #I1, err = integrate_fejer2_Xn_transformP(lambda x: 0.25/x**0.5/(1.0-x)**0.5, 0.0, 0.5, N=4.0, debug_info = True, debug_plot = True)
    #I2, err = integrate_fejer2_Xn_transformN(lambda x: 0.25/x**0.5/(1.0-x)**0.5, 0.5, 1.0, N=8, debug_info = True, debug_plot = True)
    I1, err = integrate_fejer2_Xn_transformP(ff, 0.0, 0.5, N=4.0, debug_info = True, debug_plot = True)
    print "log:", I1, err
    I2, err = integrate_fejer2_Xn_transformN(ff, 0.5, 1.0, N=4, debug_info = True, debug_plot = True)
    print "log:", I2, err, I1+I2 - pi/4

    def pc(x,t=1e10):
        #print x
        return 1/(1+(t*x)**2) * 1/(1+t**2)*t
    I, err = integrate_fejer2_pinf(pc, a=1e-10, debug_info = True, debug_plot = True)

    print "pcauchy:", I, err,


