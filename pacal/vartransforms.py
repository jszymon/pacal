"""Variable transforms.  Used for mapping to infinite intervals etc."""

from __future__ import print_function

from numpy import Inf
from numpy import hypot, sqrt, sign
from numpy import array, asfarray, empty_like, isscalar, all, equal


class VarTransform(object):
    """Base class for variable transforms."""
    def inv_var_change_with_mask(self, t):
        eq = equal.outer(t, self.var_inf)
        mask = ~eq.any(axis=-1)
        if (~mask).any():
            if isscalar(t):
                x = 0 # must masked; can pick any value, use 0
            else:
                t = asfarray(t)
                x = empty_like(t)
                x[mask] = self.inv_var_change(t[mask])
        else:
            x = self.inv_var_change(t)
        return x, mask
    def apply_with_inv_transform(self, f, t, def_val = 0, mul_by_deriv = False):
        """Apply function f to vartransform of t.

        Accepts vector inputs.  Values at infinity are set to def_val."""
        x, mask = self.inv_var_change_with_mask(t)
        if (~mask).any():
            if isscalar(x):
                y = def_val
            else:
                y = empty_like(x)
                y[mask] = f(x[mask])
                if mul_by_deriv:
                    y[mask] *= self.inv_var_change_deriv(t[mask])
                y[~mask] = def_val
        else:
            y = f(x)
            if mul_by_deriv:
                y *= self.inv_var_change_deriv(t)
        return y


class VarTransformIdentity(VarTransform):
    """The identity transform."""
    def var_change(self, x):
        return x
    def inv_var_change(self, t):
        return t
    def inv_var_change_deriv(self, t):
        return 1.0
    var_min = -1.0
    var_max = +1.0
    var_inf = [] # parameter values corresponding to infinity.  Do
                 # not distinguish +oo and -oo

### Variable transforms
class VarTransformReciprocal_PMInf(VarTransform):
    """Reciprocal variable transform."""
    def __init__(self, exponent = 1):
        self.exponent = exponent
    def var_change(self, x):
        #if x > 0:
        #    t = x / (x + 1.0)
        #else:
        #    t = x / (1.0 - x)
        t = x / (1.0 + abs(x))
        return t
    def inv_var_change(self, t):
        #if t > 0:
        #    x = t / (1.0 - t)
        #else:
        #    x = t / (1.0 + t)
        x = t / (1.0 - abs(t))
        return x
    def inv_var_change_deriv(self, t):
        return 1.0 / ((1.0 - abs(t)) * (1.0 - abs(t)))
    var_min = -1.0
    var_max = +1.0
    var_inf = [-1.0, +1.0] # parameter values corresponding to infinity.  Do
                           # not distinguish +oo and -oo
class VarTransformReciprocal_PInf(VarTransform):
    """Reciprocal variable transform.

    Optionally an exponent different from 1 can be specified.  If U is
    given, than the tranform is into finite interval [L, U]."""
    def __init__(self, L = 0, exponent = 1, U = None):
        self.exponent = exponent
        self.L = L
        self.U = U
        if self.L == 0:
            self.offset = 1.0
        else:
            self.offset = abs(self.L) / 2
        if U is not None:
            self.var_min = self.var_change(U)
            self.var_inf = []
    def var_change(self, x):
        #assert all(x >= self.L)
        if self.exponent == 1:
            t = self.offset / (x - self.L + self.offset)
        elif self.exponent == 2:
            t = sqrt(self.offset / (x - self.L + self.offset))
        else:
            t = (self.offset / (x - self.L + self.offset))**(1.0/self.exponent)
        return t
    def inv_var_change(self, t):
        if self.exponent == 1:
            x = self.L - self.offset + self.offset / t
        else:
            x = self.L - self.offset + self.offset / t**self.exponent
        return x
    def inv_var_change_deriv(self, t):
        if self.exponent == 1:
            der = self.offset / (t * t)
        else:
            der = self.offset * float(self.exponent) / t**(self.exponent + 1)
        return der
    var_min = 0
    var_max = 1
    var_inf = [0] # parameter values corresponding to infinity.  Do
                  # not distinguish +oo and -oo
class VarTransformReciprocal_MInf(VarTransform):
    """Reciprocal variable transform.

    Optionally an exponent different from 1 can be specified.  If L is
    given, than the tranform is into finite interval [L, U]."""
    def __init__(self, U = 0, exponent = 1, L = None):
        self.exponent = exponent
        self.L = L
        self.U = U
        if self.U == 0:
            self.offset = 1.0
        else:
            self.offset = abs(self.U) / 2
        if L is not None:
            self.var_min = self.var_change(L)
            self.var_inf = []
    def var_change(self, x):
        #assert all(x <= self.U)
        if self.exponent == 1:
            t = -self.offset / (x - self.U - self.offset)
        elif self.exponent == 2:
            t = sqrt(-self.offset / (x - self.U - self.offset))
        else:
            t = (self.offset / abs(x - self.U - self.offset))**(1.0/self.exponent)
        return t
    def inv_var_change(self, t):
        if self.exponent == 1:
            x = self.U + self.offset - self.offset / t
        elif self.exponent == 2:
            x = self.U + self.offset - self.offset / (t*t)
        else:
            x = self.U + self.offset - self.offset / t**self.exponent
        return x
    def inv_var_change_deriv(self, t):
        if self.exponent == 1:
            der = self.offset / (t * t)
        else:
            der = self.offset * float(self.exponent) / t**(self.exponent + 1)
        return der
    var_min = 0
    var_max = 1
    var_inf = [0] # parameter values corresponding to infinity.  Do
                  # not distinguish +oo and -oo





# variable transforms suggested by Boyd
class VarTransformAlgebraic_PMInf(VarTransform):
    """Variable transform suggested by Boyd.

    Leads to Chebyshev rational functions."""
    def __init__(self, c = 1):
        self.c = c # this corresponds to Boyd's L param
    def var_change(self, x):
        t = x / hypot(self.c, x)
        return t
    def inv_var_change(self, t):
        x = self.c * t / sqrt(1.0 - t*t)
        return x
    def inv_var_change_deriv(self, t):
        t2 = t * t
        der = t2 / sqrt((1.0 - t2)**3) + 1.0 / sqrt(1.0 - t2)
        return self.c * der
    var_min = -1.0
    var_max = +1.0
    var_inf = [-1.0, +1.0] # parameter values corresponding to infinity.  Do
                           # not distinguish +oo and -oo
class VarTransformAlgebraic_PInf(VarTransform):
    """Variable transform suggested by Boyd."""
    def __init__(self, L = 0, c = 1):
        self.L = float(L) # lower bound
        self.c = c # this corresponds to Boyd's L param
    def var_change(self, x):
        #assert all(x >= self.L)
        if ~all(x >= self.L):
            print("assert all(x >= self.L)")
            print(x)
            print(x < self.L)
        t = (x - self.L - self.c) / (x - self.L + self.c)
        return t
    def inv_var_change(self, t):
        x = self.L + self.c * (1.0 + t) / (1.0 - t)
        return x
    def inv_var_change_deriv(self, t):
        der = 2.0 * self.c / (1.0 - t)**2
        return der
    var_min = -1.0
    var_max = +1.0
    var_inf = [+1.0] # parameter values corresponding to infinity.  Do
                     # not distinguish +oo and -oo
class VarTransformAlgebraic_MInf(VarTransform):
    """Variable transform suggested by Boyd."""
    def __init__(self, U = 0, c = 1):
        self.U = float(U) # upper bound
        self.c = c # this corresponds to Boyd's L param
    def var_change(self, x):
        #assert all(x <= self.U)

        if ~all(x <= self.U):
            print("assert all(x >= self.L)")
            print(x)
            print(x < self.U)
        t = (-(x - self.U) - self.c) / (-(x - self.U) + self.c)
        return t
    def inv_var_change(self, t):
        x = self.U - self.c * (1.0 + t) / (1.0 - t)
        return x
    def inv_var_change_deriv(self, t):
        der = 2.0 * self.c / (1.0 - t)**2
        return der
    var_min = -1.0
    var_max = +1.0
    var_inf = [+1.0] # parameter values corresponding to infinity.  Do
                     # not distinguish +oo and -oo





def plot_transformed(f, vt):
    """A debugging plot of f under variable transfom vt."""
    from pylab import plot, show, linspace
    T = linspace(vt.var_min, vt.var_max, 1000)
    Y = [f(vt.inv_var_change(t)) if t not in vt.var_inf else 0 for t in T]
    plot(T, Y, linewidth=5)
def plot_transformed_w_deriv(f, vt):
    """A debugging plot of f under variable transfom vt including the
    derivative of inverse transform."""
    from pylab import plot, show, linspace
    T = linspace(vt.var_min, vt.var_max, 1000)
    Y = [f(vt.inv_var_change(t))*vt.inv_var_change_deriv(t) if t not in vt.var_inf else 0 for t in T]
    plot(T, Y, linewidth=5)
def plot_invtransformed_tail(f, vt):
    from pylab import loglog, show, logspace
    X = logspace(1, 50, 1000)
    Y = f(vt.var_change(X))
    loglog(X, Y)


if __name__ == "__main__":
    vt = VarTransformAlgebraic_PMInf()
    print(vt.inv_var_change_with_mask(array([-1,0,1])))
    print(vt.inv_var_change_with_mask(-1))
    print(vt.apply_with_inv_transform(lambda x: x+1, array([-1,0,1])))
    print(vt.apply_with_inv_transform(lambda x: x+1, 0))
    print(vt.apply_with_inv_transform(lambda x: x+1, -1))

    from numpy import exp
    from pylab import show
    #plot_transformed(lambda x: 1.0/(1+x*x), VarTransformAlgebraic_PInf(1))
    #plot_transformed(lambda x: exp(-x*x), VarTransformAlgebraic_PMInf())
    #plot_transformed_w_deriv(lambda x: 1.0/(1+x*x), VarTransformAlgebraic_PMInf())
    #plot_transformed_w_deriv(lambda x: exp(-x*x), VarTransformAlgebraic_PMInf())
    #plot_transformed(lambda x: 1.0/(1+x*x), VarTransformReciprocal_PInf())
    #plot_transformed(lambda x: exp(-x*x), VarTransformReciprocal_PInf())
    #plot_transformed(lambda x: 1.0/(1+x**1.0), VarTransformReciprocal_PInf())

    #plot_transformed(lambda x: 1.0/(1+x**1.2), VarTransformReciprocal_PInf())
    #plot_transformed(lambda x: 1.0/(1+x**1.5), VarTransformReciprocal_PInf())
    #plot_transformed(lambda x: 1.0/(1+x**2.0), VarTransformReciprocal_PInf())
    #plot_transformed(lambda x: 1.0/(1+x**2.0), VarTransformIdentity())
    #plot_transformed(lambda x: 1.0/(1+x**2.0), VarTransformReciprocal_PInf(U = 2))
    #plot_transformed(lambda x: 1.0/(1+x**2.0), VarTransformReciprocal_MInf())
    #plot_transformed(lambda x: 1.0/(1+x**2.0), VarTransformReciprocal_MInf(L = -2))

    plot_invtransformed_tail(lambda x: x, VarTransformReciprocal_PInf(L = 10))
    plot_invtransformed_tail(lambda x: 1-x, VarTransformAlgebraic_PInf(L = 10))

    show()
