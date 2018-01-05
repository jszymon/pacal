"""Multidimensional distributions."""

from __future__ import print_function

import numbers
from functools import partial
from copy import copy

import numpy
from numpy import asfarray, asmatrix, dot, delete, array, zeros, empty_like, isscalar, repeat, zeros_like, nan_to_num
from numpy import pi, sqrt, exp, argmin, isfinite, concatenate, inf
from numpy import linspace, meshgrid, transpose
from numpy.linalg import det
from pacal.utils import get_parmap

from pylab import plot, contour, xlabel, ylabel, gca, mean

import sympy as sympy

from pacal.depvars.sparse_grids import AdaptiveSparseGridInterpolator
from pacal.integration import integrate_fejer2_pminf, integrate_fejer2, integrate_iter
from pacal.segments import PiecewiseFunction

from pacal import *
from pacal.rv import RV
from pacal.segments import PiecewiseFunction
from pacal.utils import multinomial_coeff, maxprob
from pacal.integration import *
from pacal.distr import Distr


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def getRanges(vars, ci=None):
    a = zeros_like(vars)
    b = zeros_like(vars)
    for i in range(len(vars)):
        if ci is None:
            a[i], b[i] = vars[i].range()
        else:
            a[i], b[i] = vars[i].ci(ci)
    return a, b



class NDFun(object):
    # abs is a workaround for sympy.lambdify bug
    def __init__(self, d, Vars, fun, safe = False, abs = False):
        self.d = d
        self.a = [-5] * d # FIX THIS!!!
        self.b = [5] * d # FIX THIS!!!
        self.safe = safe
        self.abs = abs
        if Vars is not None:
            self.a, self.b = getRanges(Vars)
        i=0
        if Vars is None:
            i+=-1
            Vars = [RV(sym=LETTERS[i]) for i in range(d)]
        self.Vars = Vars
        var_ids = [v.id() for v in self.Vars]
        self.id_to_idx = dict((id, idx) for idx, id in enumerate(var_ids))
        self.idx_to_id = [id_ for idx, id_ in enumerate(var_ids)]
        self.symVars = [v.getSymname() for v in Vars]
        self.sym_to_var = dict((v.getSymname(), v) for v in Vars)
        if len(Vars)==1:
            a, b = getRanges(Vars)
            self.distr_pdf = PiecewiseFunction(fun=fun, breakPoints=[a[0], b[0]])
            self.fun = self.distr_pdf
        else:
            self.fun = fun
    def dump_symVars(self):
        for sv in self.Vars:
            print('{0}={1}'.format(sv.getSymname(), sv.getSym()))
    def fun(self, *X):
        pass
    def __call__(self, *X):
        if len(X) == 0:
            y = self.fun(*X)
        elif isscalar(X[0]):
            mask = (X[0] >= self.a[0]) and (X[0] <= self.b[0])
            for i in range(1, len(X)):
                mask = mask or ((X[i] >= self.a[i]) and (X[i] <= self.b[i]))
            if mask:
                y = self.fun(*X)
            else:
                y = 0
        else:
            mask = (X[0] >= self.a[0]) & (X[0] <= self.b[0])
            for i in range(1, len(X)):
                mask &= ((X[i] >= self.a[i]) & (X[i] <= self.b[i]))
            y = zeros_like(X[0])
            y[mask] = self.fun(*[xi[mask] for xi in X])
        if self.safe:
            y = nan_to_num(y)
        if self.abs:
            y = abs(y)
        return y
    def _prepare_single_var(self, var):
        if not isinstance(var, numbers.Integral):
            if var.id() in self.id_to_idx:
                var = self.id_to_idx[var.id()]
        if not isinstance(var, numbers.Integral) or not 0 <= var < self.d:
            raise RuntimeError("Incorrect marginal index " + str(var))
        return var
    def prepare_var(self, var):
        """Convert var to a list of dimension numbers."""
        if hasattr(var, "__iter__"):
            new_v = [self._prepare_single_var(v) for v in var]
        else:
            new_v = [self._prepare_single_var(var)]
        c_var = list(sorted(set(range(self.d)) - set(new_v)))
        return new_v, c_var

    def varschange(self, vari, inv_transf, inv_transf_vars, jacobian):
        #inv_transf, inv_transf_vars, jacobian = self.var_change_helper(vari, varj)
        NDinv_transf = NDDistrWithVarSubst(self, vari, inv_transf, inv_transf_vars)
        return NDProductDistr([NDinv_transf, jacobian])

#        vars_change = []
#        for i in range(len(self.Vars)):
#            if id(vari)==self.Vars[i].id():
#                vars_change.append(varj)
#            else:
#                vars_change.append(self.Vars[i])
#        return NDDistrWithVarChange(self, vars_change)
    def eliminate(self, var, a=None, b=None):
        """Integrate out var, i.e. return the marginal on all
        remaining variables.

        Optional integration limits are allowed, default being full
        real line.  var is an RV instance or dimension number"""
        var, c_var = self.prepare_var(var)
        # assume below var are dimension numbers
        if len(var) == 0:
            return self
        v1 = var[-1]
        c_var = list(sorted(set(range(self.d)) - set([v1])))
        arg = zeros(self.d)
#        def integ_f(f, arg, c_var, v1, X, x1):
#            if isscalar(x1):
#                arg[c_var] = X
#                arg[v1] = x1
#                y = f(*arg)
#            else:
#                Xcol = [None] * self.d
#                for i, j in enumerate(c_var):
#                    Xcol[j] = zeros(len(x1)) + X[i]
#                Xcol[v1] = x1
#                return f(*Xcol)
#        def interp_f(f, arg, c_var, v1, *X):
#            if isscalar(X[0]):
#                # TODO: fix integration bounds!!!!
#                if hasattr(self, "f"):
#                    y = integrate_fejer2(partial(integ_f, self, arg, c_var, v1, X, self.d), self.f.a[v1], self.f.b[v1])
#                else:
#                    y = integrate_fejer2(partial(integ_f, self, arg, c_var, v1, X, self.d), self.a[v1], self.b[v1])
#                return y[0]
#            y = asfarray(zeros_like(X[0]))
#            for i in xrange(len(X[0])):
#                # TODO: fix integration bounds!!!!
#                if hasattr(self, "f"):
#                    y[i] = integrate_fejer2(partial(integ_f, self, arg, c_var, v1, [x[i] for x in X], self.d), self.f.a[v1], self.f.b[v1])[0]
#                else:
#                    y[i] = integrate_fejer2(partial(integ_f, self, arg, c_var, v1, [x[i] for x in X], self.d), self.a[v1], self.b[v1])[0]
#            return y
        if self.d == 1:
            X = []
#            if hasattr(self, "f"):
#                y = integrate_fejer2(partial(integ_f, self.fun, arg, c_var, v1, X), self.f.a[v1], self.f.b[v1])
#            else:
#                y = integrate_fejer2(partial(integ_f, self, arg, c_var, v1, X), self.a[v1], self.b[v1])
#            return NDConstFactor(y[0])
            if hasattr(self, "f"):
                y = integrate_fejer2(partial(InterpRunner(self.fun, arg, c_var, v1).integ_f, X), self.f.a[v1], self.f.b[v1])
            else:
                y = integrate_fejer2(partial(InterpRunner(self, arg, c_var, v1).integ_f, X), self.a[v1], self.b[v1])
            return NDConstFactor(y[0])
        #m = NDInterpolatedDistr(self.d - 1, partial(self.interp_f, self, arg, c_var, v1), [self.Vars[i] for i in c_var])
        m = NDInterpolatedDistr(self.d - 1, InterpRunner(self, arg, c_var, v1).interpxx, [self.Vars[i] for i in c_var])
        if len(var) == 1:
            return m
        else:
            # TODO: eliminate all vars at once using sparse grid integration
            return m.eliminate(var[:-1])


class InterpRunner(object):
    def __init__(self, f, arg, c_var, v1):
        self.ndfun = f
        self.arg = arg
        self.c_var = c_var
        self.v1 = v1
    def interp_fx(self, X):
        ndfun = self.ndfun
        arg = self.arg
        c_var = self.c_var
        v1 = self.v1
        if hasattr(self.ndfun, "f"):
            y = integrate_fejer2(partial(self.integ_f, X), ndfun.f.a[v1], ndfun.f.b[v1])[0]
        else:
            y = integrate_fejer2(partial(self.integ_f, X), ndfun.a[v1], ndfun.b[v1])[0]
        return y
    def interpxx(self, *X):
        """convolution of f and g
        """
        ndfun = self.ndfun
        v1 = self.v1
        if isscalar(X[0]):
            # TODO: fix integration bounds!!!!
            if hasattr(self.ndfun, "f"):
                y = integrate_fejer2(partial(self.integ_f, X), ndfun.f.a[v1], ndfun.f.b[v1])
            else:
                y = integrate_fejer2(partial(self.integ_f, X), ndfun.a[v1], ndfun.b[v1])
            return y[0]
        xx = transpose(array(X))
        if isscalar(xx):
            xx=asfarray([xx])
        #print ">>>>>>>>", xx
        p_map = get_parmap()
        res = p_map(self.interp_fx, xx)
        res = array(res)
        return res

    def integ_f(self, X, x1):
        #print ":::", X, x1
        f = self.ndfun
        arg = self.arg
        c_var = self.c_var
        v1 = self.v1
        if isscalar(x1):
            arg[c_var] = X
            arg[v1] = x1
            y = f(*arg)
            #return y
        else:
            Xcol = [None] * f.d
            for i, j in enumerate(c_var):
                Xcol[j] = zeros(len(x1)) + X[i]
            Xcol[v1] = x1
            return f(*Xcol)
    def interp_f(self, *X):
        ndfun = self.ndfun
        arg = self.arg
        c_var = self.c_var
        v1 = self.v1
#        print "======="
#        print ndfun, arg, c_var, v1, X[0]
#        print self.ndfun.a
#        print self.ndfun.b
#        print self.ndfun.a[v1]
#        print self.ndfun.b[v1]
        if isscalar(X[0]):
            # TODO: fix integration bounds!!!!
            if hasattr(self.ndfun, "f"):
                y = integrate_fejer2(partial(self.integ_f, X), ndfun.f.a[v1], ndfun.f.b[v1])
            else:
                y = integrate_fejer2(partial(self.integ_f, X), ndfun.a[v1], ndfun.b[v1])
            return y[0]
        y = asfarray(zeros_like(X[0]))
        for i in range(len(X[0])):
            # TODO: fix integration bounds!!!!
            if hasattr(self.ndfun, "f"):
                y[i] = integrate_fejer2(partial(self.integ_f, [x[i] for x in X]), ndfun.f.a[v1], ndfun.f.b[v1])[0]
            else:
                y[i] = integrate_fejer2(partial(self.integ_f, [x[i] for x in X]), ndfun.a[v1], ndfun.b[v1])[0]
        return y
    def safe_interp_f(self, xt):
        #print ":::", xt
        if isscalar(xt) or len(xt)<=1:
            return self.interp_f(array([xt]))
        else:
            return self.interp_f(*xt)

def _cond_f(f, arg, c_var, var, Xcond, *Xfree):
    """Helper function for intepolating after conditional density."""
    for i, j in enumerate(var):
        arg[j] = zeros(Xfree[0].shape) + Xcond[i]
    for i, j in enumerate(c_var):
        arg[j] = Xfree[i]
    return f(*arg)
class NDDistr(NDFun):
    def __init__(self, d, Vars=None):
        super(NDDistr, self).__init__(d, Vars, self.pdf)
        self.marginals = Vars
    def condition(self, var, *X, **kwargs):
        """Return the NDDistr conditioned on var=X.

        The default implementation uses NDInterpolatedDistr"""
        var, c_var = self.prepare_var(var)
        for i in range(len(var)):
            ai, bi = self.Vars[var[i]].range()
            assert ai <= numpy.squeeze(X[i]) <= bi, "var({0})={1} outside of range [{2}, {3}]".format(var[i], X[i], ai, bi)
        if len(c_var) == 0:
            return NDOneFactor()
        if not hasattr(X, "__iter__"):
            X = [X]
        assert len(X) == len(var)
        arg = [None] * self.d
        unnormalized = NDInterpolatedDistr(len(c_var), partial(_cond_f, self, arg, c_var, var, X), [self.Vars[i] for i in c_var])
        normalize = kwargs.get("normalize", True)
        if normalize:
            nrm = unnormalized.eliminate(list(range(unnormalized.d)))
            if isinstance(nrm, NDConstFactor):
                nrm = nrm.c
            elif isinstance(nrm, NDProductDistr):
                nrm = nrm.as_constant().c
            unnormalized.f.Ys /= nrm
        return unnormalized

    def regfun(self, var, type=0):
        """It gives reggersion function E(var | I) """
        var, c_var = self.prepare_var(var)
        var, c_var  = self.Vars[var[0]], self.Vars[c_var[0]]
        #print ">>>", var, c_var
        assert self.d == 2
        def _fun(x):
            #print ">>>>>>", x, self.condition([c_var], x).distr_pdf(0.2)
            #self.condition(c_var, [x]).distr_pdf.plot()
            #show()
            if isscalar(x):
                distr = FunDistr(fun=self.condition([c_var], x).distr_pdf, breakPoints=var.get_piecewise_pdf().getBreaks())
                if type==0:
                    return distr.mean()
                elif type==1:
                    return distr.median()
                elif type==2:
                    return distr.mode()
                elif type==3:
                    return distr.quantile(0.025)
                elif type==4:
                    return distr.quantile(0.975)
                else:
                    assert 1==0
                #return distr.median()
            else:
                y =  zeros_like(x)
                for i in range(len(x)):
                    print(i, "|||", _fun(x[i]))
                    y[i] = _fun(x[i])
                return y
            #distr = FunDistr(fun=self.condition([c_var], x).distr_pdf, breakPoints=var.get_piecewise_pdf().getBreaks())
            #return distr.mean()
        #print "+++", _fun(0.5)
        return PiecewiseFunction(fun=_fun, breakPoints=c_var.get_piecewise_pdf().getBreaks()).toInterpolated()

    def cov(self, i=None, j=None):
        if i is not None and j is not None:
            var, c_var = self.prepare_var([i, j])
            dij = self.eliminate(c_var)
            f, g  = dij.Vars[0], self.Vars[1]
            fmean = 0 #f.mean()
            gmean = 0 #g.mean()
            f0, f1 = f.range()
            g0, g1 = g.range()
            if i == j:
                c, e =  1, 0#integrate_fejer2(lambda x: (x - fmean) ** 2 * f.pdf(x), f0, f1)
            else:
                c, e = integrate_iter(lambda x, y: (x - fmean) * (y - gmean) * dij.pdf(x, y), f0, f1, g0, g1)
            return c
        else:
            c = zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    #print c[i,j]
                    #print self.cov(i,j)
                    c[i, j] = self.cov(i,j)
            return c
    def mode(self):
        mo = mean(array(getRanges(self.Vars)), axis=0)
        fo = self(*[mo[i] for i in range(len(mo))])
        r = getRanges(self.Vars)
        for i in range(100):
            mi = zeros_like(mo)
            for j in range(len(r)):
                mi[j] = float(UniformDistr(r[0][j], r[1][j]).rand(1))
            fi = self(*mi)
            if fo < fi:
                mo =mi
                fo = fi
        return maxprob(self, mo*1.01, array(getRanges(self.Vars)).T)


    def _2dplot(self, n=100, cdf=False, tp = "contour", **kwargs):
        assert len(self.marginals) == 2, "Only 2d distributions can be plotted."
        if self.marginals is not None and len(self.marginals) > 1:
            f, g = self.marginals[:2]
        else:
            raise ValueError("The distribution does not have marginals")
        Lf, Uf = f.ci(0.001)
        Lg, Ug = g.ci(0.001)
        deltaf = (Uf - Lf) / n
        deltag = (Ug - Lg) / n
        X, Y = meshgrid(arange(Lf, Uf, deltaf), arange(Lg, Ug, deltag))
        if cdf:
            Z = self.cdf(X, Y)
        else:
            Z = self.pdf(X, Y)
        xlabel(f.getSymname())
        ylabel(g.getSymname())
        if tp == "contour":
            contour(X, Y, Z, 20, **kwargs)
        elif tp == "plot":
            ax = gca(projection='3d')
            ax.plot_wireframe(X, Y, Z, **kwargs)
        else:
            raise RuntimeError("Wrong plot type")
    def contour(self, n=100, cdf=False, **kwargs):
        self._2dplot(n, cdf, tp = "contour", **kwargs)
    def plot(self, n=50, cdf=False, labels = True, **kwargs):
        self._2dplot(n, cdf, tp = "plot", **kwargs)

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
        #print "========"
        #if e>1e-10:
        #    print "error L=", L, "U=", U, fun(array([U])), force_minf , force_pinf , force_poleL, force_poleU
        #    print i,e
        return i,e

class NDDistrWithVarSubst(NDDistr):
    def __init__(self, f, substvar, substfun, substfunvars):
        self.orig_f = f
        self.substfun = substfun

        # prepare variables
        substidx, _c_var = f.prepare_var(substvar)

        assert len(substidx) == 1
        self.substidx = substidx[0]
        substvar = f.Vars[self.substidx]
        newvars = list((set(f.Vars) - set([substvar])) | set(substfunvars))
        newvars.sort(key = id)
        super(NDDistrWithVarSubst, self).__init__(len(newvars), newvars)
        self.substmap_f = [(i, self.Vars.index(v)) for i, v in enumerate(f.Vars) if v != substvar]
        self.substmap_substf = [self.Vars.index(v) for v in substfunvars]
    def pdf(self, *X):
        xf = [None] * self.orig_f.d
        for i, j in self.substmap_f:
            xf[i] = X[j]
        xsubst = [X[i] for i in self.substmap_substf]
        xf[self.substidx] = self.substfun(*xsubst)
        return self.orig_f(*xf)


class Factor1DDistr(NDDistr):
    def __init__(self, distr):
        self.distr = distr
        super(Factor1DDistr, self).__init__(1, [distr])
    def __str__(self):
        return str(self.distr)
    def pdf(self, *X):
        return self.distr.pdf(*X)

class NDConstFactor(NDDistr):
    def __init__(self, c):
        super(NDConstFactor, self).__init__(0, [])
        self.c = c
    def __str__(self):
        return "const=" + str(self.c)
    def pdf(self, *X):
        assert len(X) == 0
        return self.c
    def eliminate(self, var, a=None, b=None):
        return self
    def condition(self, var, *X, **kwargs):
        return self
    def as_constant(self):
        return self.c

class NDOneFactor(NDConstFactor):
    def __init__(self):
        super(NDOneFactor, self).__init__(1)

class NDNormalDistr(NDDistr):
    def __init__(self, mu, Sigma, Vars=None):
        mu = asfarray(mu)
        Sigma = asmatrix(asfarray(Sigma))
        d = mu.shape[0]
        assert len(mu.shape) == 1
        assert len(Sigma.shape) == 2
        assert Sigma.shape[0] == Sigma.shape[1] == d
        if Vars is None:
            Vars = [NormalDistr(mu[i], Sigma[i,i], sym = LETTERS[i]) for i in range(d)]
        super(NDNormalDistr, self).__init__(d, Vars)
        self.marginals = self.Vars
        self.a = [m-5 for m in mu] # FIX THIS!!!
        self.b = [m+5 for m in mu] # FIX THIS!!!
        self.mu = mu
        self.Sigma = Sigma
        self.invSigma = array(Sigma.I)
        self.nrm = 1.0 / sqrt(det(self.Sigma) * (2 * pi) ** self.d)
    def pdf(self, *X):
        if isscalar(X) or isscalar(X[0]):
            X = asfarray(X) - self.mu
            return self.nrm * exp(-0.5 * dot(X, dot(self.invSigma, X).T))
        else:
            Xa = asfarray(X)
            Xa = Xa.transpose(list(range(1, len(X[0].shape) + 1)) + [0])
            Xa -= self.mu
            Z = (dot(Xa, self.invSigma) * Xa).sum(axis= -1)
            return self.nrm * exp(-0.5 * Z)

    def eliminate(self, var):
        var, c_var = self.prepare_var(var)
        # assume var is a dimension number here
        m_mu = delete(self.mu, var)
        m_Sigma = delete(delete(self.Sigma, var, 0), var, 1)
        if len(m_mu) == 0:
            return NDOneFactor()
        return NDNormalDistr(m_mu, m_Sigma, [self.Vars[i] for i in c_var])
    def condition(self, var, *X, **kwargs):
        var, c_var = self.prepare_var(var)
        if len(c_var) == 0:
            return NDOneFactor()
        Sigma_11 = self.Sigma[c_var, c_var]
        Sigma_12 = self.Sigma[c_var, var]
        Sigma_21 = self.Sigma[var, c_var]
        Sigma_22 = self.Sigma[var, var]
        c_mu = self.mu[c_var] + Sigma_12 * Sigma_22.I * (X - self.mu[var])
        c_mu = array(c_mu)[0]
        c_Sigma = Sigma_11 - Sigma_12 * Sigma_22.I * Sigma_21
        return NDNormalDistr(c_mu, c_Sigma, [self.Vars[i] for i in c_var])

class GausianCopula(NDDistr): # TODO
    def __init__(self, mu, Sigma, marginals=None):
        mu = asfarray(mu)
        Sigma = asmatrix(asfarray(Sigma))
        d = mu.shape[0]
        assert len(mu.shape) == 1
        assert len(Sigma.shape) == 2
        assert Sigma.shape[0] == Sigma.shape[1] == d
        Vars = marginals
        self.marginals = marginals
        if Vars is None:
            Vars = [NormalDistr(mu[i], Sigma[i,i], sym = LETTERS[i]) for i in range(d)]
        super(NDNormalDistr, self).__init__(d, Vars)
        self.marginals = self.Vars
        self.a = [m-5 for m in mu] # FIX THIS!!!
        self.b = [m+5 for m in mu] # FIX THIS!!!
        self.mu = mu
        self.Sigma = Sigma
        self.invSigma = array(Sigma.I)
        self.nrm = 1.0 / sqrt(det(self.Sigma) * (2 * pi) ** self.d)
    def pdf(self, *X):
        if isscalar(X) or isscalar(X[0]):
            X = asfarray(X) - self.mu
            return self.nrm * exp(-0.5 * dot(X, dot(self.invSigma, X).T))
        else:
            print(X)
            X = [self.marginals[i].get_piecewise_cdfinv_interp()(X[i]) for i in range(len(X))]
            Xa = asfarray(X)
            print(Xa)

            Xa = Xa.transpose(list(range(1, len(X[0].shape) + 1)) + [0])
            Xa -= self.mu
            Z = (dot(Xa, self.invSigma) * Xa).sum(axis= -1)
            return self.nrm * exp(-0.5 * Z)

    def eliminate(self, var):
        var, c_var = self.prepare_var(var)
        # assume var is a dimension number here
        m_mu = delete(self.mu, var)
        m_Sigma = delete(delete(self.Sigma, var, 0), var, 1)
        if len(m_mu) == 0:
            return NDOneFactor()
        return NDNormalDistr(m_mu, m_Sigma, [self.Vars[i] for i in c_var])
    def condition(self, var, *X, **kwargs):
        var, c_var = self.prepare_var(var)
        if len(c_var) == 0:
            return NDOneFactor()
        Sigma_11 = self.Sigma[c_var, c_var]
        Sigma_12 = self.Sigma[c_var, var]
        Sigma_21 = self.Sigma[var, c_var]
        Sigma_22 = self.Sigma[var, var]
        c_mu = self.mu[c_var] + Sigma_12 * Sigma_22.I * (X - self.mu[var])
        c_mu = array(c_mu)[0]
        c_Sigma = Sigma_11 - Sigma_12 * Sigma_22.I * Sigma_21
        return NDNormalDistr(c_mu, c_Sigma, [self.Vars[i] for i in c_var])


class NDInterpolatedDistr(NDDistr):
    def __init__(self, d, f, Vars=None, a=None, b=None):
        super(NDInterpolatedDistr, self).__init__(d, Vars)
        self.orig_f = f
#        if a is not None:
#            self.a=a
#        if b is not None:
#            self.b=b
        if Vars is not None:
            self.a, self.b = getRanges(Vars)
        #a = [-5] * d # FIX THIS!!!
        #b = [+5] * d # FIX THIS!!!
        #print "<<<<<<", Vars, self.a, self.b
        self.f = AdaptiveSparseGridInterpolator(f, d, a=self.a, b=self.b)
    def pdf(self, *X):
        return self.f.interp_at(*X)

class ConditionalDistr(NDFun):
    def __init__(self, nd, var, marg=None):
        """Condition the joint distribution nd on var."""
        super(ConditionalDistr, self).__init__(nd.d, nd.Vars, self.pdf)
        self.cond_vars, c_var = self.prepare_var(var)
        self.nd = nd
        if marg is None:
            marg = nd.eliminate(c_var)
        self.marg = marg
    def pdf(self, *X):
        return self.nd(*X) / self.marg(*[X[i] for i in self.cond_vars])
    def eliminate(self, var):
        var, c_var = self.prepare_var(var)
        if set(c_var) == set(self.cond_vars): # all unconditionned vars eliminated
            return NDOneFactor()
        if set(var) & set(self.cond_vars):
            raise RuntimeError("Cannot elimiate condition variable")
        return ConditionalDistr(self.nd.eliminate(var), [self.Vars[i] for i in self.cond_vars])
    def condition(self, var, *X, **kwargs):
        var, c_var = self.prepare_var(var)
        new_cond = set(self.cond_vars) - set(var)
        if len(new_cond) == 0: # no conditioning variables left
            return self.nd.condition(var, *X, **kwargs)
        cd = ConditionalDistr(self.nd.condition(var, *X, **kwargs), [self.Vars[i] for i in new_cond], marg=self.marg)
        return cd

def _get_str_sym_name(v):
    return str(v.getSymname())
class NDProductDistr(NDDistr):
    def __init__(self, factors):
        new_factors = []
        for f in factors:
            if isinstance(f, Distr):
                f = Factor1DDistr(f)
            new_factors.append(f)
        Vars = list(set.union(*[set(f.Vars) for f in new_factors]))
        Vars.sort(key = _get_str_sym_name) #lambda v: v.getSymname())
        super(NDProductDistr, self).__init__(len(Vars), Vars)
        self.factors = self.optimize(new_factors)
        self.a = [-inf] * len(self.Vars)
        self.b = [inf] * len(self.Vars)
        for f in self.factors:
            for i, v in enumerate(f.Vars):
                j = Vars.index(v)
                self.a[j] = max(self.a[j], f.a[i])
                self.b[j] = min(self.b[j], f.b[i])
    def __str__(self):
        s = "Factors:\n"
        for i, f in enumerate(self.factors):
            #s += "F_" + str(i) + " (" + ",".join(str(v.getSymname()) for v in f.Vars) + ")" + str(f.__class__) + "\n"
            s += "F_" + str(id(f)) + " (" + ",".join(str(v.getSymname()) for v in f.Vars) + ")" + str(f.__str__()) + "\n"
        return s
    def pdf(self, *X):
        y = 1
        for f in self.factors:
            var, c_var = self.prepare_var(f.Vars)
            #print "====", f.Vars, var, c_var
            y *= f(*[X[i] for i in var])
        return y
    def optimize(self, factors):
        # collapse constants
        new_factors = []
        c = 1
        for f in factors:
            if not isinstance(f, NDOneFactor):
                if isinstance(f, NDConstFactor):
                    c *= f.c
                else:
                    new_factors.append(f)
        if c != 1:
            new_factors.append(NDConstFactor(c))
        return new_factors
    def as_constant(self):
        """Return self as NDConstFactor if possible, else raise an exception."""
        factors = self.optimize(self.factors)
        if len(factors) == 0:
            return 1
        if len(factors) != 1 or not isinstance(factors[0], NDConstFactor):
            raise RuntimeError("Product is not constant")
        return factors[0].c

    def factor_out(self, factors, v):
        kept_factors = []
        elim_factors = []
        if hasattr(v, "__iter__"):
            v = set([vi.id() for vi in v])
        else:
            v = set([v.id()])
        for f in self.factors:
            if v & set(f.id_to_idx.keys()):
                elim_factors.append(f)
            else:
                kept_factors.append(f)
        return kept_factors, elim_factors
    def get_n_terms(self, var):
        var, c_var = self.prepare_var(var)
        assert len(var) == 1
        idv = self.Vars[var[0]].id()
        return len([f for f in self.factors if idv in f.id_to_idx])
    def eliminate(self, var, a=None, b=None):
        var, c_var = self.prepare_var(var)
        if len(var) == 0:
            return self
        Var = [self.Vars[i] for i in var]
        # heuristic to eliminate variable on which fewest factors depend
        Ns = [len([f for f in self.factors if V.id() in f.id_to_idx]) for V in Var]
        best_v = argmin(Ns)
        v1 = Var[best_v]
        # factor out
        kept_factors, elim_factors = self.factor_out(self.factors, v1)
        # do the elimination
        if len(elim_factors) == 0:
            ef = []
        elif len(elim_factors) == 1:
            ef = [elim_factors[0].eliminate(v1)]
        else:
            ef = [_NDProductDistr(elim_factors).eliminate(v1)]
        del Var[best_v]
        new_factors = self.optimize(kept_factors + ef)
        if len(new_factors) == 0:
            return NDOneFactor()
        if len(new_factors) == 1:
            return NDProductDistr([new_factors[0].eliminate(Var)])
        newpd = NDProductDistr(new_factors)
        if len(Var) == 0:
            return newpd
        else:
            return newpd.eliminate(Var)
    def condition(self, var, *X, **kwargs):
        var, c_var = self.prepare_var(var)
        Var = [self.Vars[i] for i in var]
        # factor out
        kept_factors, cond_factors = self.factor_out(self.factors, Var)
        new_cond_factors = []
        for cf in cond_factors:
            cV = []; cX = []
            for i, V in enumerate(Var):
                if V in cf.Vars:
                    cV.append(V)
                    cX.append(X[i])
            #print cf, cf.Vars[0].getSymname(), cf.condition(cV, cX, normalize=False)
            new_cond_factors.append(cf.condition(cV, cX, normalize=False))
        cfp = NDProductDistr(kept_factors + new_cond_factors)
        nrm = cfp.eliminate(list(range(cfp.d)))
        nrm = nrm.as_constant()
        if nrm == 0:
            nrm = 1
        cfp.factors = cfp.optimize(cfp.factors + [NDConstFactor(1.0 / nrm)])
        return cfp
    #def varschange(self, vari, varj):
    def varschange(self, vari, inv_transf, inv_transf_vars, jacobian):
        #inv_transf, inv_transf_vars, jacobian = self.var_change_helper(vari, varj)
        vari, _c_var = self.prepare_var(vari)
        vari = self.Vars[vari[0]]
        kept_factors, elim_factors = self.factor_out(self.factors, vari)
        new_factors = kept_factors
        for ef in elim_factors:
            new_factors.append(NDDistrWithVarSubst(ef, vari, inv_transf, inv_transf_vars))
        new_factors.append(jacobian)
        #print "kept factors=", kept_factors
        #print "elim factors=", elim_factors
        #print "new factors=", new_factors
        return NDProductDistr(new_factors)

class _NDProductDistr(NDProductDistr):
    """A temporary class for elimination after factoring variables
    out."""
    def eliminate(self, var, a=None, b=None):
        return NDDistr.eliminate(self, var, a, b)

class IJthOrderStatsNDDistr(NDDistr):
    """return joint p.d.f. of the i-th and j-th order statistics
    for sample size of n, 1<=i<j<=n, see springer's book p. 347
    """
    def __init__(self, f, n, i, j):
        #print n, i, j, (1 <= i and i < j and j <= n)
        assert (1 <= i and i < j and j <= n), "incorrect values 1<=i<j<=n"
        self.f = f.get_piecewise_pdf()
        self.F = f.get_piecewise_cdf().toInterpolated()
        self.S = (1.0 - self.F).toInterpolated()
        X = iid_order_stat(f, n, i, sym="X_{0}".format(i))
        Y = iid_order_stat(f, n, j, sym="X_{0}".format(j))
        Vars = [X, Y]
        super(IJthOrderStatsNDDistr, self).__init__(2, Vars)
        self.i = i
        self.j = j
        self.n = n
        a, b = getRanges([f, f])
        self.a = a
        self.b = b
    def pdf(self, *X):
        assert len(X) == 2, "this is 2d distribution"
        x, y = X
        i, j, n = self.i, self.j, self.n
        mask = x < y
        return mask * multinomial_coeff(n, [i - 1, 1, j - i - 1, 1, n - j]) * self.F(x) ** (i - 1.0) * (self.F(y) - self.F(x)) ** (j - i - 1) * (1.0 - self.F(y)) ** (n - j) * self.f(x) * self.f(y)

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
        #print "========"
        #if e>1e-10:
        #    print "error L=", L, "U=", U, fun(array([U])), force_minf , force_pinf , force_poleL, force_poleU
        #    print i,e
        return i,e




def plot_2d_distr(f, theoretical=None, have_3d = False, cont_levels=10):
    # plot distr in 3d
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    #try:
    #    have_3d = True
    #    ax = fig.add_subplot(111, projection='3d')
    #except:
    #    ax = fig.add_subplot(111)
    #    have_3d = False
    #have_3d = True
    a, b = f.a, f.b
    #a, b = getRanges(f.Vars)
    a, b = getRanges(f.Vars, ci=0.0001)
    #print "a, b = ", a, b
    X = np.linspace(a[0], b[0], 100)
    Y = np.linspace(a[1], b[1], 100)
    X, Y = np.meshgrid(X, Y)
    #XY = np.column_stack([X.ravel(), Y.ravel()])
    #Z = asfarray([f(xy) for xy in XY])
    #Z.shape = (X.shape[0], Y.shape[0])
    Z = f(X, Y)
    #print "==", f,Z, (X<Y)
    if theoretical is not None:
        Zt = theoretical(X, Y)
    if theoretical is not None:
        fig = plt.figure()
    fig = plt.gcf()

    if have_3d:
        #ax = fig.add_subplot(111, projection='3d')
        ax = plt.gca(projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', antialiased=True)
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
        ax.set_xlabel(f.Vars[0].getSymname())
        ax.set_ylabel(f.Vars[1].getSymname())
        if theoretical is not None:
            plt.figure()
            ax = plt.gca(projection='3d')
            ax.plot_surface(X, Y, Z - Zt, rstride=1, cstride=1)
    else:
        #ax = fig.add_subplot(111)
        nc = cont_levels
        ax = plt.gca()
        maxV = numpy.max(Z)
        minV = max(0, numpy.min(Z))
        dV = (maxV-minV)/nc
        V = numpy.round(linspace(minV + dV/8, maxV-dV/2, nc)*100)/100.0
        #print "max=", maxV, V
        C = ax.contour(X, Y, Z, V, colors="k") #)#),
        C.clabel()
        #fig.colorbar(C)
        ax.set_xlabel(f.Vars[0].getSymname())
        ax.set_ylabel(f.Vars[1].getSymname())
        if theoretical is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            C = ax.contour(X, Y, Z - Zt)
            fig.colorbar(C)

def plot_1d1d_distr(free_distr, a, b, fun):
    # plot distr in 3d
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    X = np.linspace(a, b, 1000)
    Y = fun(X)
    Z = free_distr(X)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = plt.gca(projection='3d')
    #ax = plt.gca()
    ax.plot_surface(np.vstack([X,X]), np.vstack([Y,Y]), np.vstack([np.zeros_like(Z),Z]),
                    cstride = 1, rstride = 1,# cmap=cm.jet,
                    linewidth = -1, edgecolor="b", color = "b", alpha=1, antialiased = False)
    return ax

class NDNoisyFun(NDDistr):
    """Function with additive noise.

    Behaves like a conditional distribution."""
    def __init__(self, f, f_vars, noise_distr = BetaDistr(5, 5), value_sym = "fun_value", f_range = None):
        a, b = getRanges(f_vars)  # just set to infinity?
        if f_range is None:
            # assume f is monotonic in all vars.  TODO: fix this!
            f_a = f(*a)
            f_b = f(*b)
            f_a, f_b = min(f_a, f_b), max(f_a, f_b)
        else:
            f_a, f_b = f_range
        noise_range = getRanges([noise_distr])
        noise_a, noise_b = noise_range[0][0], noise_range[1][0]
        a = concatenate([a, [noise_a + f_a]])
        b = concatenate([b, [noise_b + f_b]])
        Vars = f_vars + [RV(sym = value_sym, a = a[-1], b = b[-1])]
        d = len(Vars)
        super(NDNoisyFun, self).__init__(d, Vars)
        self.fun_value_var = Vars[-1]
        self.f = f
        self.f_range = f_range
        self.noise_distr = noise_distr
        self.a, self.b = a, b
        self.noise_a, self.noise_b = noise_a, noise_b

    def pdf(self, *X):
        if isscalar(X[0]):
            z = X[-1]
            fy = self.f(*X[:-1])
            if self.noise_a <= z - fy <= self.noise_b:
                y = self.noise_distr(z - fy)
            else:
                y = 0
        else:
            y = zeros_like(X[0])
            z = X[-1]
            fy = self.f(*X[:-1])
            mask = (self.noise_a <= z - fy) & (z - fy <= self.noise_b)
            y[mask] = self.noise_distr((z - fy)[mask])
        return y

    def eliminate(self, var):
        var, c_var = self.prepare_var(var)
        if len(var) != 1 or var[0] != self.d - 1:
            raise RuntimeError("Can only eliminate the function value variable")
        return NDOneFactor()
    class partial_f(object):
        def __init__(self, f, d, inds, vals):
            self.orig_f = f
            self.orig_d = d
            assert len(inds) == len(vals)
            self.inds = inds
            self.vals = vals
            self.var_inds = list(sorted(set(range(d)) - set(inds)))
        def __call__(self, *X):
            args = [None] * self.orig_d
            for i, v in zip(self.inds, self.vals):
                args[i] = zeros_like(X[0]) + v
            for i, x in zip(self.var_inds, X):
                args[i] = x
            return self.orig_f(*args)
    def condition(self, var, *X, **kwargs):
        var, c_var = self.prepare_var(var)
        pf = NDNoisyFun.partial_f(self, self.d, var, X)
        new_vars = [self.Vars[i] for i in c_var]
        cd = NDFun(self.d - 1, new_vars, pf)
        return cd

if __name__ == "__main__":


    from pylab import *
    from pacal import *
    #from pacal.depvars.copulas import *

    X = BetaDistr(3,3, sym = "X")
    Y = BetaDistr(3,4, sym = "Y")
    noise = BetaDistr(5,5)*2 - 1
    print(X, Y, noise)
    nf = NDNoisyFun(lambda x, y: x + y, [X, Y], noise, value_sym = "Z")
    #nf = NDNoisyFun(lambda x, y: x * y, [X, Y], noise, value_sym = "Z")
    print(nf.a, nf.b)
    pr = NDProductDistr([X, Y, nf])
    print(pr.a, pr.b)
    zd = pr.eliminate([X])
    plot_2d_distr(zd)
    figure()
    zd = zd.eliminate([Y])
    pfun = FunDistr(zd, breakPoints = [zd.a[0], zd.b[0]])
    pfun.plot()
    pfun.summary()
    ylim(ymin = 0)
    ylabel("$PDF_Z$")
    zd2 = pr.eliminate([nf.fun_value_var])
    plot_2d_distr(zd2)
    zd3 = pr.condition([nf.fun_value_var], 0)
    plot_2d_distr(zd3)
    xlim(0,0.7)
    ylim(0,0.65)
    show()
    0/0

    params.interpolation.maxn = 10
    params.interpolation.use_cheb_2nd = False
    X, Y = UniformDistr() + UniformDistr(), BetaDistr(1,4)  #BetaDistr(6,1)
    #c = ClaytonCopula(theta = 0.5, marginals=[X, Y])
    #c = ClaytonCopula(theta = 5, marginals=[X, Y])
    c = FrankCopula(theta = 3, marginals=[X, Y])
    #c = GumbelCopula(theta = 2, marginals=[X, Y])
    plot_2d_distr(c)
    fun0 = c.regfun(Y, type=0)
    fun0.plot(linewidth=2.0, color='r', label="mean")
    fun1 = c.regfun(Y, type=1)
    fun1.plot(linewidth=2.0, color='g', label="median")
    fun2 = c.regfun(Y, type=2)
    fun2.plot(linewidth=2.0, color='b', label="mode")

    fun3 = c.regfun(Y, type=3)
    fun3.plot(linewidth=1.0, color='k', label="ci_L")
    fun4 = c.regfun(Y, type=4)
    fun4.plot(linewidth=1.0, color='k', label="ci_U")
    legend()
    rx,ry = c.rand2d_invcdf(500)
    plot(rx,ry,'.')
    c.plot()
    show()
    0/0

    #from pacal.depvars.copulas import *
    #c = ClaytonCopula(theta = 0.5, marginals=[UniformDistr(), UniformDistr()])

    d = IJthOrderStatsNDDistr(BetaDistr(2,2), 10, 1, 10)
    print(d.symVars)
    plot_2d_distr(d)
    show()
    from pylab import figure, plot, linspace, show, legend
#    mu = [0,0,0]
#    Sigma = [[1, 0.5, 0.7], [0.5, 1, 0.4], [0.7, 0.4, 1]]
#    ndn = NDNormalDistr(mu, Sigma)
#    print ndn.cov(1,2)
#    0/0
    X = linspace(-5, 5, 1000)

    ### multivariate normal
    mu = [1, 2]
    Sigma = [[2, 0.5], [0.5, 1]]
    ndn = NDNormalDistr(mu, Sigma)
    print(ndn(0, 0))
    print(ndn.Vars)

    ndn_sub = NDDistrWithVarSubst(ndn, ndn.Vars[0], lambda x,y: x+y, [ndn.Vars[0], ndn.Vars[1]])
    print(ndn(1, 1), ndn_sub(0,1))

    plot_2d_distr(ndn)

    ndn_marg = ndn.eliminate(1)
    ndn_marg = ndn.eliminate(ndn.Vars[1])
    print(ndn_marg.Vars)
    figure()

    plot(X, [ndn_marg(x) for x in X])

    ndn_cond = ndn.condition(ndn.Vars[1], 0.5)
    print(ndn_cond.Vars)
    figure()
    plot(X, [ndn_cond(x) for x in X])


    ndi = NDInterpolatedDistr(ndn.d, ndn)
    print(ndi.Vars)

    plot_2d_distr(ndi)

    ndi_int = ndi.eliminate([0, 1])
    print(ndi_int)

    ndi_marg = ndi.eliminate(ndi.Vars[1])
    print(ndi_marg.Vars)
    figure()
    plot(X, [ndi_marg(x) for x in X])
    plot(X, [ndn_marg(x) for x in X])

    ndi_cond = ndi.condition(ndi.Vars[1], 0.5)
    figure()
    plot(X, [ndi_cond(x) for x in X])
    plot(X, [ndn_cond(x) for x in X])

    ndn_cond2 = ndn.condition(ndn.Vars[1], 2)
    ndi_cond2 = ndi.condition(ndi.Vars[1], 2)
    ndn_cond3 = ConditionalDistr(ndn, 1)
    figure()
    plot(X, ndi_cond2(X), label="Interp")
    plot(X, ndn_cond2(X), label="Normal")
    plot(X, [ndn_cond3(x, 2) for x in X], label="ConditionalDistr")
    legend()

    ### Gaussian MRF example
    mu = [0, 0]
    Sigma1 = [[1, 0.5], [0.5, 1]]
    Sigma2 = [[1, 0.75], [0.75, 1]]
    ndn1 = NDNormalDistr(mu, Sigma1)
    ndn2 = NDNormalDistr(mu, Sigma2, Vars=[ndn1.Vars[0], RV()])
    ndn1c = ConditionalDistr(ndn1, [0])
    ndn2c = ConditionalDistr(ndn2, [0])
    gmrf = NDProductDistr([ndn1c, ndn2c, ndn1.eliminate(1)])
    print(gmrf(0, 0, 0))

    print(ndn1.eliminate([0, 1]))

    plot_2d_distr(ndn1)
    plot_2d_distr(ndn2)

    gmmarg1 = gmrf.eliminate(ndn2.Vars[1])
    plot_2d_distr(gmmarg1)

    gmmarg2 = gmrf.eliminate(ndn1.Vars[1])
    plot_2d_distr(gmmarg2)

    gmmarg3 = gmrf.eliminate(ndn2.Vars[0])
    plot_2d_distr(gmmarg3)

    gmmarg4 = gmrf.eliminate([ndn1.Vars[1], ndn2.Vars[1]])
    figure()
    plot(X, gmmarg4(X))

    gmmarg5 = gmrf.eliminate([0, 1, 2])
    print(gmmarg5.as_constant())

    gmrfc1 = gmrf.condition(ndn1.Vars[0], 1)
    plot_2d_distr(gmrfc1)  # [1, 2] are conditionally independent given [0]

    gmrfc2 = gmrf.condition(ndn1.Vars[1], 1)
    plot_2d_distr(gmrfc2)
    gmrfc3 = gmrf.condition(ndn1.Vars[1], -1)
    plot_2d_distr(gmrfc3)

    show()
