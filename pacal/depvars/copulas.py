"""Set of copulas different types"""

from __future__ import print_function

from pacal.integration import *
from pacal.interpolation import *
from matplotlib.collections import PolyCollection

import pacal.distr
#from pacal import *
from pacal.segments import PiecewiseDistribution, MInfSegment, PInfSegment, Segment, _segint
from pacal.segments import PiecewiseFunction

from pacal.distr import Distr
from pacal.standard_distr import *
#from pacal.nddistr import NDDistr, NDInterpolatedDistr, NDFun

from pacal.utils import epsunique, bisect, fmin2

from pacal.indeparith import _findSegList, convdiracs
from pacal.integration import integrate_fejer2, integrate_iter

from pacal.depvars.nddistr import NDDistr, NDFun
import pylab as plt

import sympy
import numpy as np
from sympy import Symbol, diff, pprint, simplify

from pylab import meshgrid, contour, xlabel, ylabel, gca, figure, axis
import mpl_toolkits.mplot3d.axes3d as p3


try:
    from scipy.optimize.optimize import fminbound
    have_Scipy_optimize = True
except ImportError:
    have_Scipy_optimize = False



class Copula(NDDistr):
    def __init__(self, marginals=None):
        self.marginals = marginals
        super(Copula, self).__init__(len(self.marginals), Vars=self.marginals)
        self.a, self.b = self.ranges()

    def ranges(self):
        vars = self.marginals
        a = zeros_like(vars)
        b = zeros_like(vars)
        for i in range(len(vars)):
            a[i], b[i] = vars[i].range()
        return a, b
    def setMarginals(self, *marginals):
        if len(marginals) > 0 and isinstance(marginals[0], pacal.distr.Distr):
            self.marginals = marginals
    def pdf(self, *X):
        """joint probability density function with marginals *X"""
        if self.marginals is None or len(self.marginals) == 0:
            U = UniformDistr()
            F = [U.get_piecewise_cdf_interp()(X[i]) for i in range(len(X))]
            return self.cpdf(*F)
        else:
            #assert len(self.marginals) >= len(X)
            mi = ones_like(X[0])
            for i in range(len(X)):
                mi = mi * self.marginals[i].get_piecewise_pdf()(X[i])
            F = [self.marginals[i].get_piecewise_cdf_interp()(X[i]) for i in range(len(X))]
            return np.nan_to_num(self.cpdf(*F) * mi)
            #return self.cpdf(*F) * mi
    def cdf(self, *X):
        """joint cumulative distribution function with given marginals at point (x,y)"""
        if self.marginals is None or len(self.marginals) == 0:
            return self.ccdf(*X)
        else:
            F = [self.marginals[i].get_piecewise_cdf_interp()(X[i]) for i in range(len(X))]
            return self.ccdf(*F)

    def dualcdf(self, *X):
        si = zeros_like(X[0])
        for i in range(len(X)):
            si += self.marginals[i].get_piecewise_cdf_interp()(X[i])
        return si - self.ccdf(*X)
    def jpdf_(self, f, g, x, y):
        """joint probability density function with marginals *X"""
        if isinstance(f, Distr):
            return self.cpdf(f.get_piecewise_cdf_interp()(x), g.get_piecewise_cdf_interp()(y)) * f.get_piecewise_pdf()(x) * g.get_piecewise_pdf()(y) 
        else:
            return self.cpdf(f.cumint()(x), g.cumint()(y)) * f(x) * g(y)
    def jcdf_(self, f, g, x, y):
        """joint cumulative distribution function with marginals f, g at point (x,y)"""
        #return self.ccdf(f.get_piecewise_cdf()(X), g.get_piecewise_cdf()(Y))
        return self.ccdf(f.get_piecewise_cdf()(x), g.get_piecewise_cdf()(y))
    def cpdf(self, *X):
        """Copula density, joint probability density function with uniform U[0,1] marginals"""
        #pass
        pass #return zeros_like(X[0])
    def ccdf(self, *X):
        """Copula, joint cumulative distribution function  with uniform  U[0,1] marginals"""
        pass
    def debug_plot(self, n=40, show_pdf=False, azim=210, elev=30):
        #Z = self.cdf(f.get_piecewise_cdf()(X), g.get_piecewise_cdf()(Y))
        #Z = self.jcdf(f, g, X, Y)
        if self.marginals is not None and len(self.marginals) > 1:
            f, g = self.marginals[:2]
            self.setMarginals((f, g))
        else:
            f, g = UniformDistr(), UniformDistr()

        Lf, Uf = f.ci(0.01)
        Lg, Ug = g.ci(0.01)
        deltaf = (Uf - Lf) / n
        deltag = (Ug - Lg) / n
        X, Y = meshgrid(arange(Lf, Uf, deltaf), arange(Lg, Ug, deltag))
        if not show_pdf:
            Z = self.cdf(X, Y)
            fig = figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)
            #ax = p3.Axes3D(fig)
            xf = arange(Lf, Uf, deltaf)
            xg = arange(Lg, Ug, deltag)
            cf = f.cdf(xf)
            cg = g.cdf(xg)
            ax.plot(xf, cf, zs=Ug, zdir='y', linewidth=3.0, color="k")
            ax.plot(xg, cg, zs=Uf, zdir='x', linewidth=3.0, color="k")
            ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', antialiased=True)#cmap=cm.jet
            cset = ax.contour(X, Y, Z, zdir='z', color='k', offset=0)
            ax.set_xlabel('$X$')
            ax.set_xlim3d(Lf, Uf)
            ax.set_ylabel('$Y$')
            ax.set_ylim3d(Lg, Ug)
            ax.set_zlabel('$Z$')
            ax.set_zlim3d(0, 1)
        else:
            fig = figure(figsize=plt.figaspect(1))
            ax2 = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)
            Z2 = self.pdf(X, Y)
            xf = arange(Lf, Uf, deltaf)
            xg = arange(Lg, Ug, deltag)
            cf = f.pdf(xf)
            cg = g.pdf(xg)
            ax2.plot(xf, cf, zs=Ug, zdir='y', linewidth=3.0, color="k")
            ax2.plot(xg, cg, zs=Uf, zdir='x', linewidth=3.0, color="k")
            ax2.plot_wireframe(X, Y, Z2, rstride=1, cstride=1, color='k', antialiased=True)
            cset = ax2.contour(X, Y, Z2, color='k', zdir='z', offset=0)
            ax2.set_xlabel('$X$')
            ax2.set_xlim3d(Lf, Uf)
            ax2.set_ylabel('$Y$')
            ax2.set_ylim3d(Lg, Ug)
            ax2.set_zlabel('$Z$')
            zlim = 1.01*np.max(array([np.max(Z2), max(cf), max(cg)]))
            ax2.set_zlim3d(0,zlim)


    def _segint(self, fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
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
        return i,e

    def cov(self, i=None, j=None):
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            f, g = dij.marginals[0], self.marginals[1]
            fmean = f.mean()
            gmean = g.mean()
            f0, f1 = f.get_piecewise_pdf().range()
            g0, g1 = g.get_piecewise_pdf().range()
            print(fmean, gmean, var, c_var, f0, f1, g0, g1)
            if i == j:
                c, e = c, e = integrate_fejer2(lambda x: (x - fmean) ** 2 * f.pdf(x), f0, f1)                  
            else:
                c, e = integrate_iter(lambda x, y: (x - fmean) * (y - gmean) * dij.pdf(x, y), f0, f1, g0, g1)
            return c
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    c[i, j] = self.cov(i, j)                                          
            return c
    def corrcoef(self, i=None, j=None):
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            f, g = dij.marginals[0], self.marginals[1]
            return self.cov(i, j)/f.std()/g.std()
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    c[i, j] = self.corrcoef(i, j)                                          
            return c
    def tau(self, i=None, j=None):
        """Kendall's tau: 4*\int C(x,y) dC(x,y)-1
        """
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            f, g = dij.marginals[0], self.marginals[1]
            f0, f1 = f.get_piecewise_pdf().range()
            g0, g1 = g.get_piecewise_pdf().range()
            if i == j:
                c, e = 1, 0
            else:
                c, e = integrate_iter(lambda x, y: dij.cdf(x, y) * dij.pdf(x, y), f0, f1, g0, g1)
                c = 4 * c - 1
            return c
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    c[i, j] = self.ctau(i, j)                                          
            return c
    def beta(self, i=None, j=None):
        """Blomqvist's beta: 4 * C(0.5, 0.5) - 1
        """
        return 4*self.ccdf(0.5,0.5)-1
    def rho_s(self, i=None, j=None):
        """Spearmans rho: 12*\int x*y dC(x,y)-3 = 12 \int C(d,y)dxdy - 3
        """
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            if i == j:
                c, e = 1, 0
            else:
                #c, e = integrate_iter(lambda x, y: x * y * dij.cpdf(x, y), 0.0, 1.0, 0.0, 1.0)
                c, e = integrate_iter(lambda x, y: dij.ccdf(x, y), 0.0, 1.0, 0.0, 1.0)
                c = 12 * c - 3
            return c
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    c[i, j] = self.rho_s(i, j)                                          
            return c

    def ctau(self, i=None, j=None):
        """Kendall's tau: 4*\int C(x,y) dC(x,y)-1
        """
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            if i == j:
                c, e = 1, 0
            else:
                #c, e = integrate_iter(lambda x, y: x * y * dij.cpdf(x, y), 0.0, 1.0, 0.0, 1.0)
                c, e = integrate_iter(lambda x, y: dij.ccdf(x, y) * dij.cpdf(x, y), 0.0, 1.0, 0.0, 1.0)
                c = 4 * c - 1
            return c
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    c[i, j] = self.ctau(i, j)                                          
            return c

class PiCopula(Copula):
    def __init__(self, marginals=None):
        super(PiCopula, self).__init__(marginals=marginals)
    def cpdf(self, *X):
        return ones_like(X[0])
    def ccdf(self, *X):
        pi = ones_like(X[0])
        for xi in X:
            pi *= xi
        return pi
class MCopula(Copula):
    def __init__(self, marginals=None):
        super(MCopula, self).__init__(marginals)
        self._segint = self._segmin
    def cpdf(self, *X):
        return zeros_like(X[0])#self.ccdf(*X)
    def ccdf(self, *X):
        mi = zeros_like(X[0])+1
        for xi in X[0:]:
            xia = array(xi)
            ind = xia < mi
            if isscalar(mi) | size(mi)==1:
                mi = xia
            else:
                mi[ind] = xia[ind]
        return mi
    def _segmin(self, fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
            debug_info = False, debug_plot = False):
        xopt = fmin2(lambda x: fun(float(x)), L, U, xtol = 1e-16)
        return xopt, 0#fun(xopt), 0
    def debug_plot(self, n=40, show_pdf=False, azim=210, elev=30):
        #Z = self.cdf(f.get_piecewise_cdf()(X), g.get_piecewise_cdf()(Y))
        #Z = self.jcdf(f, g, X, Y)
        if self.marginals is not None and len(self.marginals) > 1:
            f, g = self.marginals[:2]
            self.setMarginals((f, g))
        else:
            f, g = UniformDistr(), UniformDistr()

        Lf, Uf = f.ci(0.01)
        Lg, Ug = g.ci(0.01)
        deltaf = (Uf - Lf) / n
        deltag = (Ug - Lg) / n

        X, Y = meshgrid(arange(Lf, Uf, deltaf), arange(Lg, Ug, deltag))
        if not show_pdf:
            Z = self.cdf(X, Y)
            Z2 = self.pdf(X, Y)
            fig = figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)
            #ax = p3.Axes3D(fig)
            xf = arange(Lf, Uf, deltaf)
            xg = arange(Lg, Ug, deltag)
            cf = f.cdf(xf)
            cg = g.cdf(xg)
            ax.plot(xf, cf, zs=Ug, zdir='y', linewidth=3.0, color="k")
            ax.plot(xg, cg, zs=Uf, zdir='x', linewidth=3.0, color="k")
            cset = ax.contour(X, Y, Z, zdir='z', offset=0)
            ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', antialiased=True)#cmap=cm.jet
            ax.set_xlabel('$X$')
            ax.set_xlim3d(Lf, Uf)
            ax.set_ylabel('$Y$')
            ax.set_ylim3d(Lg, Ug)
            ax.set_zlabel('$Z$')
            ax.set_zlim3d(0, 1)
            # wykres F(x)=G(Y)
        else:
            fig = figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d')
            #ax = fig.add_subplot(122,  projection='3d')
            t = linspace(0.01, 0.99,40)
            X = f.quantile(t)
            Y = g.quantile(t)
            Z = f(X)*g(Y)
            cf = f.pdf(xf)
            cg = g.pdf(xg)
            ax.plot(xf, cf, zs=Ug, zdir='y', linewidth=3.0, color="k")
            ax.plot(xg, cg, zs=Uf, zdir='x', linewidth=3.0, color="k")
            ax.plot_surface(np.vstack([X,X]), np.vstack([Y,Y]), np.vstack([np.zeros_like(Z),Z]),
                        cstride = 1, rstride = 1,# cmap=cm.jet,
                        linewidth = -1, edgecolor="k", color = "c", alpha=0.7, antialiased = True)
            ax.axis((Lf, Uf, Lg, Ug))
            zlim = 1.01*np.max(array([max(Z), max(cf), max(cg)]))
            ax.set_zlim3d(0,zlim)

class WCopula(MCopula):
    def __init__(self, marginals=None):
        super(WCopula, self).__init__(marginals)
        self._segint = self._segmax
    def cpdf(self, *X):
        return zeros_like(X[0])#self.ccdf(*X)
    def ccdf(self, *X):
        si = zeros_like(X[0])
        for xi in X[0:]:
            si += array(xi)
        si = si - 1
        ind = (si < 0)
        if isscalar(si) | size(si)==1:
            if ind:
                si = 0.0
        else:
            si[ind] = 0
        return si
    def _segmax(self, fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
            debug_info = False, debug_plot = False):
        #xopt = fminbound(fun, L, U, xtol = 1e-16)
        #xopt = fminbound(lambda x: 100-fun(float(x)), L, U, xtol = 1e-16)
        xopt = fmin2(lambda x: 1-fun(float(x)), L, U, xtol = 1e-16)
        return xopt, 0
    def _segmin(self, fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
            debug_info = False, debug_plot = False):
        #xopt = fminbound(fun, L, U, xtol = 1e-16)
        #xopt = fminbound(lambda x: 100-fun(float(x)), L, U, xtol = 1e-16)
        xopt = fmin2(lambda x: fun(float(x)), L, U, xtol = 1e-16)
        return xopt, 0

class ArchimedeanCopula(Copula):
    # TODO
    def __init__(self, fi=log, fi_deriv=lambda s: 1 / s,
                 fi_inv=exp, fi_inv_nth_deriv=exp,
                 marginals=None):
        super(ArchimedeanCopula, self).__init__(marginals)
        #self.theta = Symbol('theta')
        self.fi = fi
        self.fi_deriv = fi_deriv
        self.fi_inv = fi_inv
        self.fi_inv_nth_deriv = fi_inv_nth_deriv
        #self.debug_info_()
    def debug_info_(self):
        vars = self.symVars
        #for i in range(self.d):
        #    vars.append(sympy.Symbol("u{0}".format(i + 1)))
        si = 0
        for i in range(self.d):
            si += self.fi(vars[i])
        pi = 1;
        for i in range(self.d):
            pi *= self.fi_deriv(vars[i])
        print("si=\n", pprint(si))
        print("pi=\n", pprint(pi))
        print("C=\n", pprint(self.fi_inv(si)))
        #print "C=\n", pprint(self.ccdf(*tuple(vars)))
        #print "c=\n", pprint(sympy.simplify(self.fi_inv_nth_deriv(si) * pi))
        print("c=\n", pprint(self.fi_inv_nth_deriv(si) * pi))
        #print "c=\n", pprint(self.cpdf(*tuple(vars)))
    def tau_c(self):
        return 1 + 4 * integrate_fejer2(lambda t : self.fi(t) / self.fi_deriv(t), 0, 1)[0]

    def cpdf(self, *X):
        assert len(X) == len(self.marginals), "incorrect copula dimension"
        si = zeros_like(X[0])
        for xi in X:
            si = si + self.fi(xi)
        si = self.fi_inv_nth_deriv(si)
        pi = ones_like(X[0])
        for xi in X:
            pi = pi * self.fi_deriv(xi)
        return si * pi
    def ccdf(self, *X):
        assert len(X) == len(self.marginals), "incorrect copula dimension"
        si = zeros_like(X[0])
        for xi in X:
            si += self.fi(xi)
        ind = (si < 0) # or isnan(si)
        #if len(ind)>0:
        si[ind] = 0.0
        si = self.fi_inv(si)
        return si
class ArchimedeanSymbolicCopula(ArchimedeanCopula):
    # TODO
    def __init__(self,
                 fi=lambda t, theta: log(t),
                 fi_inv=None, #lambda t, theta:(-sympy.log(t)) ** theta,
                 theta=2,
                 marginals=None):
        self.theta = float(theta)#Symbol('theta')
        self.t = Symbol('t')
        self.s = Symbol('s')
        self.d = len(marginals)
        self.fi_ = fi
        self.fi_inv_ = fi_inv
        self.sym_fi = fi(self.t, self.theta)
        self.sym_fi_deriv = sympy.diff(self.sym_fi, self.t)
        if fi_inv is  None:
            self.sym_fi_inv = sympy.solve(self.sym_fi - self.s, self.t)[0]
        else:
            self.sym_fi_inv = fi_inv(self.s, self.theta)
        self.sym_fi_inv_nth_deriv = sympy.diff(self.sym_fi_inv, self.s, self.d)
        #self.debug_info()
        super(ArchimedeanSymbolicCopula, self).__init__(fi=sympy.lambdify(self.t, self.sym_fi, "numpy"),
                                                        fi_deriv=sympy.lambdify(self.t, self.sym_fi_deriv, "numpy"),
                                                        fi_inv=sympy.lambdify(self.s, self.sym_fi_inv, "numpy"),
                                                        fi_inv_nth_deriv=sympy.lambdify(self.s, self.sym_fi_inv_nth_deriv, "numpy"),
                                                        marginals=marginals)
        vars = self.symVars
        si = 0
        for i in range(self.d):
            si += self.fi_(vars[i], self.theta)
        self.sym_C = self.fi_inv_(si, self.theta)

    def eliminate(self, var):
        var, c_var = self.prepare_var(var)
        c_marginals = [self.marginals[i] for i in c_var]
        if len(var) == 0:
            return self
        return ArchimedeanSymbolicCopula(fi=self.fi_,
                                         fi_inv=self.fi_inv_,
                                         theta=self.theta,
                                         marginals=c_marginals)

    def ccond(self, var):
        """It returns conditional copula f([var, c_vars]) = C(c_var | var)
        """
        var, c_var = self.prepare_var(var)
        symvars = [self.symVars[i] for i in var]
        DC = self.sym_C
        for i in range(len(self.Vars)):
            if i in set(var):
                DC = sympy.diff(DC, self.symVars[i])
            else:
                pass
        dC = sympy.lambdify(self.symVars, DC, "numpy")
        return NDFun(self.d, self.Vars, sympy.lambdify(self.symVars, DC, "numpy"))       

    def condition(self, var, *X):
        """It returns conditional pdf for given copula
        f(c_var) = Pr(c_var | var=X)
        """
        var, c_var = self.prepare_var(var)
        num = self.pdf
        den = self.eliminate(c_var)
        def fun_(*Y_):
            j, k = 0, 0
            Y, Yvar = [], []
            #dF = ones_like(X[0])
            for i in range(len(self.Vars)):
                if i in set(var):
                    Y.append(X[j])
                    Yvar.append(X[j])
                    j += 1
                else:
                    Y.append(Y_[k])
                    k += 1
            return num(*Y) / den.pdf(*X)
        return NDFun(len(c_var), [self.Vars[i] for i in c_var], fun_)
    def conditionCDF(self, var, *X):
        """It returns conditional cdf for given copula
        f(c_var) = Pr(Y<c_var | var=X)
        """
        funcond = self.ccond(var)
        var, c_var = self.prepare_var(var)
        new_cond = var
        def fun_(*Y_):
            j, k = 0, 0
            Y = []
            dF = ones_like(X[0])
            for i in range(len(self.Vars)):
                if i in set(var):
                    Y.append(self.marginals[i].get_piecewise_cdf()(X[j]))
                    j += 1
                else:
                    Y.append(self.marginals[i].get_piecewise_cdf()(Y_[k]))
                    dF *= self.marginals[i].get_piecewise_pdf()(Y_[k])
                    k += 1
            return funcond(*Y)
        return NDFun(len(new_cond), [self.Vars[i] for i in c_var], fun_)
    def condfun(self, var):
        """It returns conditional cdf function f([var, c_vars]) = Pr(Y<c_var | var)
        """
        funcond = self.ccond(var)
        var, c_var = self.prepare_var(var)
        new_cond = var
        def fun_(*X):
            j, k = 0, 0
            Y = []
            dF = ones_like(X[0])
            for i in range(len(self.Vars)):
                Y.append(self.marginals[i].get_piecewise_cdf()(X[i]))
                if i in set(var):
                    pass
                else:
                    dF *= self.marginals[i].get_piecewise_pdf()(X[i])                    
            return funcond(*Y)
        return NDFun(self.d, self.Vars, fun_)
    def debug_info(self):
        #self.fi_inv_defiv = simplify(sympy.diff(self.sym_fi_inv(self.s, self.theta), self.s))
        print("theta=", self.theta)
        print("fi(theta)=", self.fi_(self.t, sympy.Symbol("theta")))
        print("fi=\n", pprint(self.sym_fi))
        print("fi_deriv=\n", pprint(self.sym_fi_deriv))
        print("fi_inv=\n", self.sym_fi_inv, ",\n", pprint(self.sym_fi_inv))
        print("fi_inv_nth_deriv=\n", pprint(self.sym_fi_inv_nth_deriv))
        print("fi=\n", sympy.latex(self.sym_fi))
        print("fi_deriv=\n", sympy.latex(self.sym_fi_deriv))
        print("fi_inv=\n", self.sym_fi_inv, ",\n", sympy.latex(self.sym_fi_inv)) 
        print("fi_inv_nth_deriv=\n", sympy.latex(self.sym_fi_inv_nth_deriv))
    def rand2d_invcdf(self, n):
        u = self.marginals[0].rand_invcdf(n)
        t = UniformDistr().rand(n)
        v = zeros_like(t)
        for i in range(len(u)):
            #Cd = self.condition([0],u[i])
            #print i
            v[i] = self.conditionCDF([0], u[i]).distr_pdf.inverse(t[i])
            #v[i] = bisect(lambda x : condition(x,u[i])-t[i], 1e-50,1)
        return u, v


class GumbelCopula2d(Copula):
    def __init__(self, theta=3.1, marginals=None):
        super(GumbelCopula2d, self).__init__(marginals)
        self.theta = theta
        self.one_over_theta = 1.0 / theta
        self.theta_square = theta ** 2
    def fi(self, t):
        return pow(-np.log(t), self.theta)# ** self.theta
    def fi_inv(self, s):
        return exp(-s ** self.one_over_theta)
    def cpdf(self, *X):
        si = zeros_like(X[0])
        for xi in X:
            si += self.fi(xi)
        si = self.fi_inv(si) * (si ** (self.one_over_theta - 2.0) * (-1.0 + self.theta + si ** self.one_over_theta)) / self.theta_square
        for xi in X:
            si *= self.theta * self.fi(xi) ** (1 - self.one_over_theta) / xi          
        return si
    def ccdf(self, *X):
        si = zeros_like(X[0])
        for xi in X:
            si += self.fi(xi)
        si = self.fi_inv(si)
        return si

class GumbelCopula(ArchimedeanSymbolicCopula):
    """Clayton copula, C(theta=-1) = W, C(theta=0) = Pi, C(theta=+Inf) = M"""
    def __init__(self, theta=3.1, marginals=None):
        super(GumbelCopula, self).__init__(fi=self.fi_, fi_inv=self.fi_inv_,
                                            theta=theta, marginals=marginals)
    def fi_(self, t, theta):
        return (-sympy.log(t)) ** theta
    def fi_inv_(self, s, theta):
        return sympy.exp(-(s ** (1 / theta)))


class ClaytonCopula(ArchimedeanSymbolicCopula):
    """Clayton copula, C(theta=-1) = W, C(theta=0) = Pi, C(theta=+Inf) = M"""
    def __init__(self, theta=3.1, marginals=None):
        super(ClaytonCopula, self).__init__(fi=self.fi_, fi_inv=self.fi_inv_,
                                            theta=theta, marginals=marginals)
#        theta = float(theta)
#        self.theta = theta
#        self.one_over_theta = 1.0 / theta
#        self.theta_square = theta ** 2
    def fi_(self, t, theta):
        return 1 / theta * (t ** (-theta) - 1)
#        #return self.one_over_theta * (pow(t, -self.theta) - 1.0)
    def fi_inv_(self, s, theta):
        return (1 + s * theta) ** (-1 / theta)
#    def cpdf(self, *X):
#        si = zeros_like(X[0])
#        for xi in X:
#            si += xi ** -self.theta
#        si = si - 1
#        ind = (si < 0) # or isnan(si)
#        si[ind] = 0
#        si = (1 + self.theta) / self.theta_square * si ** (-(self.one_over_theta + 2)) 
#        for xi in X:
#            si *= -self.theta * xi ** (-self.theta - 1)
#        return si
#    def ccdf(self, *X):
#        si = zeros_like(X[0])
#        for xi in X:
#            si += self.fi(xi)
#        ind = si < 0
#        si[ind] = 0
#        si = self.fi_inv(si)
#        return si
class FrankCopula(ArchimedeanSymbolicCopula):
    """Clayton copula, C(theta=-1) = W, C(theta=0) = Pi, C(theta=+Inf) = M"""
    def __init__(self, theta=3.1, marginals=None):
        self.const2 = exp(-theta) - 1.0
        super(FrankCopula, self).__init__(fi=self.fi_, fi_inv=self.fi_inv_,
                                          theta=theta, marginals=marginals)

    def fi_(self, t, theta):
        return -sympy.log((sympy.exp(-t * theta) - 1) / (exp(-self.theta) - 1.0))
#    def fi_(self, t, theta):
#        return - log((exp(-t * theta) - 1) / (exp(-self.theta) - 1.0))
    def fi_inv_(self, s, theta):
        return -sympy.log(sympy.exp(-s - theta) - sympy.exp(-s) + 1) / theta

class FrankCopula2d(Copula):
    """Frank copula, C(theta=-Inf) = W, C(theta=0)~Pi, C(theta=+Inf)=M
    B3 in H. Joe pp. 139-
    """
    def __init__(self, theta=1.0, marginals=None):
        self.theta = theta # delta
        self.eta = -expm1(-self.theta)
        self.one_over_theta = 1.0 / theta
        self.theta_square = theta ** 2
        super(FrankCopula2d, self).__init__(marginals)
    def fi(self, t):
        return logexp_m1(t * self.theta) - logexp_m1(self.theta)
    def fi_inv(self, s):
        if expm1(-self.theta) > 0:
            return -1.0 / self.theta * logexp_p1(-s, expm1(-self.theta))
        elif expm1(-self.theta) < 0:
            return -1.0 / self.theta * log_1m_exp(-s, expm1(-self.theta))
        else:
            return -1.0 / self.theta * logexp_p1(-s, 0)
    def cpdf(self, *X):
        si = zeros_like(X[0])
        pi = ones_like(X[0])
        n = len(X)
        for xi in X:
            si += xi
            pi *= -np.expm1(-self.theta * xi)
        yi = self.theta * self.eta * np.exp(-self.theta * si) / (self.eta - pi) ** n
        return yi
    def ccdf(self, *X):
        pi = ones_like(X[0])
        for xi in X:
            pi *= -expm1(-self.theta * xi)
        yi = -self.one_over_theta * np.log1p(-pi / self.eta)
        return yi


def logexp_p1(x, a=1.0):
    """return log(a*exp(x) + 1)"""
    x = x + log(abs(a))
    yy = log1p(exp(x))
    ind = exp(x) > 1e16
    yy[ind] = x[ind]
    ind = exp(x) < 1e-16
    yy[ind] = exp(x[ind])
    return yy
def logexp_m1(x, a=1.0):
    """return -log(a*exp(-x) - 1)"""
    x = x + log(abs(a))
    yy = -log(abs(expm1(-x)))
    if isscalar(x):
        if exp(-x) > 1e16:
            yy = x
        if exp(-x) < 1e-16:
            yy = exp(-x)
    else:
        ind = exp(-x) > 1e16
        yy[ind] = x[ind]
        ind = exp(-x) < 1e-16
        yy[ind] = exp(-x[ind])
    return yy
def log_1m_exp(x, a=1.0):
    """return -log(1-a*exp(-x))"""
    x = x + log(abs(a))
    yy = log(abs(expm1(x)))
    ind = exp(x) > 1e16
    yy[ind] = x[ind]
    ind = exp(x) < 1e-16
    yy[ind] = -exp(x[ind])
    return yy


def convmean(F, G, p=0.5, q=0.5, theta=1.0):
    """Probabilistic weighted mean of f and g
    """
    f = F.get_piecewise_pdf()
    g = G.get_piecewise_pdf()
    if  p + q != 1.0 :
        p1 = abs(p) / (abs(p) + abs(q))
        q = abs(q) / (abs(p) + abs(q))
        p = p1;
    if q == 0:
        return f;
    bf = f.getBreaks()
    bg = g.getBreaks()
    b = add.outer(bf * p, bg * q)
    fun = lambda x : convmeanx(F, G, segList, x, p, q, theta=theta)
    ub = epsunique(b)
    fg = PiecewiseDistribution([]);
    op = lambda x, y : p * x + q * y;
    if isinf(ub[0]):
        segList = _findSegList(f, g, ub[1] - 1, op)
        seg = MInfSegment(ub[1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub = ub[1:]
    if isinf(ub[-1]):
        segList = _findSegList(f, g, ub[-2] + 1, op)
        seg = PInfSegment(ub[-2], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)
        ub = ub[0:-1]
    for i in range(len(ub) - 1) :
        segList = _findSegList(f, g, (ub[i] + ub[i + 1]) / 2, op)
        seg = Segment(ub[i], ub[i + 1], fun)
        segint = seg.toInterpolatedSegment()
        fg.addSegment(segint)

    # Discrete parts of distributions
    fg_discr = convdiracs(f, g, fun=lambda x, y : x * p + y * q)
    for seg in fg_discr.getDiracs():
        fg.addSegment(seg)
    return fg

def convmeanx(F, G, segList, xx, p=0.5, q=0.5, theta=2):
    """Probabilistic weighted mean of f and g, integral at points xx
    """
    if size(xx) == 1:
        xx = asfarray([xx])
    wyn = zeros_like(xx)
    #P = PiCopula()
    #P = GumbelCopula(theta)
    P = FrankCopula2d(theta)
    #P.corrcoef()
    #P = ClaytonCopula(theta)
    #fun = lambda t : P.fun(segi( t / p)/q, segj((x - t)/q)/q)
    #W = PiCopula()
    #fun = lambda t : P.ccdf(segi(t / p) / p / q, segj((x - t) / q) / p / q)
    fun = lambda t : P.jpdf(F, G, (t / p), (x - t) / q) / p / q


    for j in range(len(xx)) :
        x = xx[j]
        I = 0
        err = 0
        for segi, segj in segList:
            if segi.isSegment() and segj.isSegment():
                L = max(segi.a * p, (x - segj.b * q))
                U = min(segi.b * p, (x - segj.a * q))
                i, e = _segint(fun, L, U)
            #elif segi.isDirac() and segj.isSegment():
            #    i = segi.f*segj((x-segi.a)/q)/q   # TODO
            #    e=0;
            #elif segi.isSegment() and segj.isDirac():
            #    i = segj.f*segi((x-segj.a)/p)/p   # TODO
            #    e=0;
            #elif segi.isDirac() and segj.isDirac():
            #    pass
            #    #i = segi(x-segj.a)/p/q          # TODO
            #    #e=0;
            I += i
            err += e
        wyn[j] = I
    return wyn
if __name__ == "__main__":
    from pylab import *
    from .nddistr import plot_2d_distr
#    # ========= ArchimedeanCopulas tests ============================
#    A = ArchimedeanSymbolicCopula(fi=lambda t, theta : 1 / theta * (t ** (-theta) - 1),
#                          #fi_inv=lambda s, theta : (1+ theta*s) ** (-1/theta),
#                          theta=1.0,
#                          marginals=[BetaDistr(4, 4, sym="X"), BetaDistr(2, 4, sym="Y"), BetaDistr(5, 3, sym="Z")])
#    #BetaDistr(2, 3).summary()
    from pacal.depvars.nddistr import *
    c = ClaytonCopula(theta = 0.2, marginals=[UniformDistr(), UniformDistr()])
    c.plot()
    d = IJthOrderStatsNDDistr(UniformDistr(), 10, 1, 10)
    plot_2d_distr(d)
    show()
    0/0

    marginals = [BetaDistr(5, 2, sym="X"), BetaDistr(3, 6, sym="Y")]

    C = FrankCopula(10, marginals)
    C.plot()
    plot_2d_distr(C)
    C_condition_y_05 = C.condition([1], 0.5)
    figure()
    C_condition_y_05.distr_pdf.plot()
    #print C_condition_y_05.distr_pdf.summary()
    show()
    0 / 0
