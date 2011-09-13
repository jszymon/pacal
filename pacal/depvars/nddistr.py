'''
Created on 12-09-2011

@author: marcin
'''
import numbers

from pylab import *
from pacal.segments import PiecewiseFunction
from pacal.stats.iid_ops import iid_order_stat 
from pacal.utils import multinomial_coeff
from pacal.integration import *
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def getRanges(vars):
    a = zeros_like(vars)
    b = zeros_like(vars)
    for i in range(len(vars)):
        a[i], b[i] = vars[i].range()
    return a, b

class NDFun(object):
    # abs is a workaround for sympy.lambdify bug
    def __init__(self, d, Vars, fun, safe = False, abs = False):
        assert Vars is not None, "Vars cannot be None"
        self.d = d
        self.a = [-5] * d # FIX THIS!!!
        self.b = [5] * d # FIX THIS!!!
        self.safe = safe
        self.abs = abs
        if Vars is not None:
            self.a, self.b = getRanges(Vars)
        self.Vars = Vars
        var_ids = [id(v) for v in self.Vars]
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
            print '{0}={1}'.format(sv.getSymname(), sv.getSym())
    def fun(self, *X):
        pass
    def __call__(self, *X):
        y = self.fun(*X)
        if self.safe:
            y = nan_to_num(y)
        if self.abs:
            y = abs(y)
        return y
    def _prepare_single_var(self, var):
        if id(var) in self.id_to_idx:
            var = self.id_to_idx[id(var)]
        if not isinstance(var, numbers.Integral) or not 0 <= var < self.d:
            print "!!!", var
            raise RuntimeError("Incorrect marginal index")
        return var
    def prepare_var(self, var):
        """Convert var to a list of dimension numbers."""
        if hasattr(var, "__iter__"):
            new_v = [self._prepare_single_var(v) for v in var]
        else:
            new_v = [self._prepare_single_var(var)]
        c_var = list(sorted(set(range(self.d)) - set(new_v)))
        return new_v, c_var


class NDDistr(NDFun):
    def __init__(self, d, Vars=None):
        super(NDDistr, self).__init__(d, Vars, self.pdf)  

class IJthOrderStatsNDDistr(NDDistr):
    """return joint p.d.f. of the i-th and j-th order statistics
    for sample size of n, 1<=i<j<=n, see springer's book p. 347 
    """
    def __init__(self, f, n, i, j):
        print n,i,j,(1<=i and i<j and j<=n)
        assert (1<=i and i<j and j<=n), "incorrect values 1<=i<j<=n" 
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
        
        a, b = getRanges([f ,f])
        self.a = a
        self.b = b
    def pdf(self, *X):
        assert len(X)==2, "this is 2d distribution"
        x, y = X
        i, j, n = self.i, self.j, self.n
        mask  = x<y
        #print "===========", n, i, j
        #print mask  
        #print multinomial_coeff(n, [i-1,1, j-i-1,1, n-j]) 
        #print  self.F(x)**(i-1.0) 
        #print  (self.F(y) - self.F(x))**(j-i-1)
        #print  (1.0-self.F(y))**(n-j)
        return mask * multinomial_coeff(n, [i-1,1, j-i-1,1, n-j]) * self.F(x)**(i-1.0) * (self.F(y) - self.F(x))**(j-i-1) * (1.0-self.F(y))**(n-j)
    def _segint(self, fun, L, U, force_minf = False, force_pinf = False, force_poleL = False, force_poleU = False,
                debug_info = False, debug_plot = False):
        #print params.integration_infinite.exponent
        if L > U:
            if params.segments.debug_info:
                print "Warning: reversed integration interval, returning 0"
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
            print "errors in _conv_div: x, segi, segj, L, U =", L, U
        #print "========"
        #if e>1e-10:
        #    print "error L=", L, "U=", U, fun(array([U])), force_minf , force_pinf , force_poleL, force_poleU
        #    print i,e
        return i,e     
def plot_2d_distr(f, theoretical=None):
    # plot distr in 3d
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()    
    #try:
    #    have_3d = True
    #    ax = fig.add_subplot(111, projection='3d')
    #except:
    #    ax = fig.add_subplot(111)
    #    have_3d = False
    have_3d = True
    a, b = f.a, f.b
    print "a, b = ", a, b
    X = np.linspace(a[0], b[0], 100)
    Y = np.linspace(a[1], b[1], 100)
    X, Y = np.meshgrid(X, Y)
    #XY = np.column_stack([X.ravel(), Y.ravel()])
    #Z = asfarray([f(xy) for xy in XY])
    #Z.shape = (X.shape[0], Y.shape[0])
    Z = f(X, Y)
    print Z* (X<Y)
    if theoretical is not None:
        Zt = theoretical(X, Y)
    if theoretical is not None:
        fig = plt.figure()
        
    if have_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', antialiased=True)
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
        ax.set_xlabel(f.Vars[0].getSymname())
        ax.set_ylabel(f.Vars[1].getSymname())
        if theoretical is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot_surface(X, Y, Z - Zt, rstride=1, cstride=1)
    else:
        ax = fig.add_subplot(111)
        C = ax.contour(X, Y, Z, 100)
        fig.colorbar(C)
        ax.set_xlabel(f.Vars[0].getSymname())
        ax.set_ylabel(f.Vars[1].getSymname())
        if theoretical is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            C = ax.contour(X, Y, Z + Zt)
            fig.colorbar(C)

if __name__ == "__main__":
    from pylab import *
    from pacal import *
    #from pacal.depvars.copulas import *
    #c = ClaytonCopula(theta = 0.5, marginals=[UniformDistr(), UniformDistr()])
    
    d = IJthOrderStatsNDDistr(UniformDistr(), 10, 3, 7)
    print d.symVars
    plot_2d_distr(d)
    show()