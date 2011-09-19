"""Random Variable class"""

import numbers
from functools import partial

import numpy
from numpy import array, zeros_like, unique, concatenate, isscalar, isfinite
from numpy import sqrt, pi, arctan, tan, asfarray
from numpy.random import uniform
from numpy import minimum, maximum, add, subtract, divide, prod

from numpy.lib.function_base import histogram
from numpy import hstack
from pylab import bar

from sympy import var, log, exp
import sympy

from pacal.utils import Inf

import traceback

class RV(object):
    def __init__(self, parents = [], sym = None, a=-5, b=5):
        self.parents = parents
        self.a = a
        self.b = b
        if sym is not None:         # ====set symbolic name of r.v
            if isinstance(sym, basestring):
                self.sym = sympy.Symbol(sym) 
            else:
                self.sym = sym      # users defined symbolic name of r.v.                            
        else:
            self.sym = var("X{0}".format(id(self))) # default symbolic name of r.v.
        if self.sym.is_Atom:
            #print "atom: ", self.sym
            self.symname = self.sym
        else:
            #print "complex: ", self.sym
            self.symname = sympy.Symbol("X{0}".format(id(self)))
    def __str__(self):
        return "RV(" + str(self.sym) + ")"
    
    def __repr__(self):
        return self.__str__()   
    
    def range(self):
        return self.a, self.b
    
    def getId(self):
        return id(self)
    
    def getName(self):
        """return, string representation of RV"""
        return type
    def getSym(self):
        """return, symbolic representation of RV"""
        return self.sym              
    
    def setSym(self, sym):
        """it set, symbolic name of RV"""
        if isinstance(sym, str):
            self.symname = sympy.Symbol(sym) 
        else:
            self.symname = sym                            
    def getSymname(self):
        return self.symname
                
    def getEquations(self, node=None, l=[], r=[]):
        if node is None:
            node = self
        p = node.parents
        if not node.getSym().is_Atom and not node.getSym() in set(l):
            l.append(node.getSym()) 
            r.append(node.getSymname())
        if p is None or len(p)==0:
            return l, r
        elif len(p)==1: 
            l, r = self.getEquations(p[0], l,r)
            return l, r
        else:
            l, r = self.getEquations(p[0], l,r)
            l, r = self.getEquations(p[1], l,r)
            return l, r
#    def getSymFree(self):
#        def _get_rv(rootrv, sym):
#            if isinstance(sym, str):
#                sym = sympy.Symbol(sym) 
#            rvs  = rootrv.getParentsAll()
#            print rvs
#            for rv in rvs:
#                print "---", rv.getSymname(), sym, rv.getSymname()
#                if rv.getSymname() == sym:
#                    return rv
#            return None
#             
#        l, r  = self.getEquations()
#        for var in r:
#            free  = self.getParentsFree()
#            linked  = self.getParentsDep()
#            print "=====",var
#            print free
#            print linked
#        print "====="
#        print r
#        print l
#        print _get_rv(self, "x")
        
#    def clear(self, node=None):
#        if node is None:
#            node = self
#        p = node.parents
#        print node.getSym()
#        if p is None or len(p)==0:
#            print "+", node.getSym()
#            return
#        elif len(p)==1: 
#            self.clear(p[0])   
#            print "-", p[0].getSym()         
#        elif len(p)==2:
#            self.clear(p[0])
#            self.clear(p[1])
#            print "--", p[0].getSym()
#            print "--", p[1].getSym()
#        #del node            
#        return
                                
    def getOperation(self):
        """return string representation of operation when is used, None otherwise."""
        return None

    def _addvar(self, newv):
        for v in self.vars:
            if v == newv: return # taka zmienna juz jest
        self.vars.append(newv)
    def isFree(self):
        return len(self.parents)==0
    def isLinked(self):
        return len(self.parents)>0
    def getParentsAll(self, var=None, lista=set()):
        if var is None:
            var=self
        if not var in lista:
                lista.add(var)
        for p in var.parents:            
            self.getParentsDep(p, lista) 
        return lista
    def getParentsFree(self, var=None, lista=set()):
        if var is None:
            var=self
        if len(var.parents)==0:
            if not var in lista:
                lista.add(var)
        for p in var.parents:            
            self.getParentsDep(p, lista) 
        return lista
    def getParentsDep(self, var=None, lista=set()):
        if var is None:
            var=self
        if len(var.parents)>0:
            if var not in lista:
                lista.add(var)
        for p in var.parents:            
            self.getParentsDep(p, lista) 
        return lista
    
    # overload arithmetic operators
    def __neg__(self):
        """Overload negation distribution of -X."""
        return ShiftedScaledRV(self, scale = -1)
    def __abs__(self):
        """Overload abs: distribution of abs(X)."""
        return AbsRV(self)
    #def __sign__(self):
    #    """Overload sign: distribution of sign(X)."""
    #    return SignDistr(self)
    def __add__(self, d):
        """Overload sum: distribution of X+Y."""
        if isinstance(d, RV):
            return SumRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, shift = d)
        raise NotImplemented()
    def __radd__(self, d):
        """Overload sum with real number: distribution of X+r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, shift = d)
        raise NotImplemented()
    def __sub__(self, d):
        """Overload subtraction: distribution of X-Y."""
        if isinstance(d, RV):
            return SubRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, shift = -d)
        raise NotImplemented()
    def __rsub__(self, d):
        """Overload subtraction with real number: distribution of X-r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, scale = -1, shift = d)
        raise NotImplemented()
    def __mul__(self, d):
        """Overload multiplication: distribution of X*Y."""
        if isinstance(d, RV):
            return MulRV(self, d)
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledRV(self, scale = d)
        raise NotImplemented()
    def __rmul__(self, d):
        """Overload multiplication by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledRV(self, scale = d)
        raise NotImplemented()
    def __div__(self, d):
        """Overload division: distribution of X*r."""
        if isinstance(d, RV):
            return DivRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, scale = 1.0 / d)
        raise NotImplemented()
    def __rdiv__(self, d):
        """Overload division by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            #return FuncRV(self, lambda x: d/x, lambda x: d/x, lambda x: d/x**2)
            return d * InvRV(self)
        raise NotImplemented()
    def __pow__(self, d):
        """Overload power: distribution of X**Y, 
        and special cases: X**(-1), X**2, X**0. X must be positive definite."""        
        if isinstance(d, RV):
            return ExpRV(MulRV(LogRV(self), d))
        if isinstance(d, numbers.Real):
            if d == 0:
                return 1
            elif d == 1:
                return self
            elif d == -1:
                return InvRV(self)
            elif d == 2:
                return SquareRV(self)
            else:
                return ExpRV(ShiftedScaledRV(LogRV(self), scale = d))
                #return PowRV(self, alpha = d)
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
            return ExpRV(ShiftedScaledRV(self, scale = numpy.log(x)))
        raise NotImplemented()

    
class OpRV(RV):
    """Base class for operations on distributions.

    Currently only does caching for random number generation."""
    def __str__(self):
        #return "min(#{0}, #{1})".format(id(self.d1), id(self.d2))
        if len(self.parents)==2:
            if isinstance(self.getSym(), sympy.Add):
                op = "+"
            if isinstance(self.getSym(), sympy.Mul):
                op = "*"
            else: self.getSym().__class__
            return "({0}{2}{1})".format(self.parents[0], self.parents[1], op)
class FuncRV(OpRV):
    """Injective function of random variable"""
    def __init__(self, d, f, f_inv, f_inv_deriv, pole_at_zero = False, fname = "f", sym = None):
        super(FuncRV, self).__init__([d], sym = sym)
        self.d = d
        self.f = f
        self.f_inv = f_inv
        self.f_inv_deriv = f_inv_deriv
        self.fname = fname
    def __str__(self):
        #return "{0}(#{1})".format(self.fname, id(self.d))
        return "{0}({1})".format(self.fname, self.d)
    def getName(self):
        return "{0}({1})".format(self.fname, self.d.getName())
    def getOperation(self):
        """return string representation of operation when is used, None otherwise."""
        return self.fname

class ShiftedScaledRV(OpRV):
    def __init__(self, d, shift = 0, scale = 1):
        assert(scale != 0)
        self.shift = shift
        self.scale = scale
        super(ShiftedScaledRV, self).__init__([d], sym = (d.getSymname()*self.scale+self.shift))
        self.d = d
        self._1_scale = 1.0 / scale
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
        elif self.shift == 0:
            return "{0}*{1}".format(self.scale, self.d.getName())
        elif self.scale == 1:
            return "{0}{1:+}".format(self.d.getName(), self.shift)
        else:
            return "({2}*{0}+{1})".format(self.d.getName(), self.shift, self.scale)

class ExpRV(FuncRV):
    """Exponent of a random variable"""
    def __init__(self, d):
        super(ExpRV, self).__init__(d, numpy.exp, numpy.log,
                                       lambda x: 1.0/abs(x), pole_at_zero = True, 
                                       fname = "exp", sym = sympy.exp(d.getSymname()))
    def is_nonneg(self):
        return True
def exp(d):
    """Overload the exp function."""
    if isinstance(d, RV):
        return ExpRV(d)
    return numpy.exp(d)
class LogRV(FuncRV):
    """Natural logarithm of a random variable"""
    def __init__(self, d):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        super(LogRV, self).__init__(d, numpy.log, numpy.exp,
                                       numpy.exp, pole_at_zero= True, 
                                       fname = "log", sym = sympy.log(d.getSymname()))
    
def log(d):
    """Overload the log function."""
    if isinstance(d, RV):
        return LogRV(d)
    return numpy.log(d)

def sign(d):
    """Overload sign: distribution of sign(X)."""
    if isinstance(d, RV):
        return SignRV(d)
    return numpy.sign(d)
class AtanRV(FuncRV):
    """Arcus tangent of a random variable"""
    def __init__(self, d):
        super(AtanRV, self).__init__(d, numpy.arctan, self.f_inv,
                                        self.f_inv_deriv, pole_at_zero= False,
                                        fname ="atan", sym = sympy.atan(d.getSymname()))
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
    if isinstance(d, RV):
        return AtanRV(d)
    return numpy.arctan(d)

class InvRV(OpRV):
    """Inverse of random variable."""
    def __init__(self, d):
        super(InvRV, self).__init__([d], sym = 1/d.getSymname())
        self.d = d
        self.pole_at_zero = False
    def __str__(self):
        return "1/#{0}".format(id(self.d))    
    def getName(self):
        return "1/{0}".format(self.d.getName())    
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

class PowRV(FuncRV):
    """Inverse of random variable."""
    def __init__(self, d, alpha = 1):
        super(PowRV, self).__init__([d],self.f_, self.f_inv, self.f_inv_deriv, pole_at_zero = alpha > 1, fname="pow", sym = d.getSymname()**alpha)
        self.d = d
        self.alpha = alpha
        self.alpha_inv = 1.0 / alpha
        self.exp_deriv = self.alpha_inv - 1.0
    def __str__(self):
        return "1/#{0}".format(id(self.d))    
    def getName(self):
        return "{0}^{1}".format(self.d.getName(), self.alpha)    
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
    
class AbsRV(OpRV):
    """Absolute value of a distribution."""
    def __init__(self, d):
        super(AbsRV, self).__init__([d], sym = sympy.abs(d.getSymame())) # TODO abs unpleasant in sympy
        self.d = d
    def __str__(self):
        return "|#{0}|".format(id(self.d))
    def getName(self):
        return "|{0}|".format(self.d.getName())

class SignRV(RV):
    def __init__(self, d):
        self.d = d
        super(SignRV, self).__init__(d, sym = sympy.sign(d.getSymname()))
    #def getSym(self):
    #    raise NotImplemented()
    def __str__(self):
        return "sign({0})".format(id(self.d))
    def getName(self):
        return "sign({0})".format(self.d.getName())
        
class SquareRV(OpRV):
    """Injective function of random variable"""
    def __init__(self, d):
        super(SquareRV, self).__init__([d], sym = d.getSymname()**2)
        self.d = d
    def __str__(self):
        return "#{0}**2".format(id(self.d))
    def getName(self):
        return "sqr({0})".format(self.d.getName())

def sqrt(d):
    if isinstance(d, RV):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        return d ** 0.5
    return numpy.sqrt(d)

class SumRV(OpRV):
    """Sum of distributions."""
    def __init__(self, d1, d2):
        super(SumRV, self).__init__([d1, d2], sym = d1.getSymname() + d2.getSymname())
        breaks = unique(add.outer(d1.range(), d2.range()))
        self.a, self.b = min(breaks), max(breaks)
        self.d1 = d1
        self.d2 = d2
    def getName(self):
        #return "({0}+{1})".format(self.d1.getName(), self.d2.getName())
        return "({0}+{1})".format(self.d1, self.d2)
    def getOperation(self):
        return "+"
class SubRV(OpRV):
    """Difference of distributions."""
    def __init__(self, d1, d2):
        super(SubRV, self).__init__([d1, d2], sym = d1.getSymname() - d2.getSymname())
        breaks = unique(subtract.outer(d1.range(), d2.range()))
        self.a, self.b = min(breaks), max(breaks)
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        # return "#{0}-#{1}".format(id(self.d1), id(self.d2))
        return "({0}-{1})".format(self.d1, self.d2)
    def getName(self):
        return "({0}-{1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "-"

class MulRV(OpRV):
    def __init__(self, d1, d2):
        super(MulRV, self).__init__([d1, d2], sym = d1.getSymname() * d2.getSymname())
        breaks = unique(multiply.outer(d1.range(), d2.range()))
        self.a, self.b = min(breaks), max(breaks)
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        #return "#{0}*#{1}".format(id(self.d1), id(self.d2))
        return "({0}*{1})".format(self.d1, self.d2)
    def getName(self):
        return "({0}*{1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "*"
    
class DivRV(OpRV):
    def __init__(self, d1, d2):
        super(DivRV, self).__init__([d1, d2], sym = d1.getSymname() / d2.getSymname())
        d1range = list(d1.range()) 
        d2range = list(d2.range())
        if prod(d1range)<0:
            d1range.append(0.0)
        if prod(d2range)<0:
            d2range.append(0.0)          
        breaks = unique(divide.outer(d1range, d2range))
        self.a, self.b = min(breaks), max(breaks)
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        # return "#{0}/#{1}".format(id(self.d1), id(self.d2))
        return "{0}/{1}".format(self.d1, self.d2)
    def getName(self):
        return "({0}/{1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "/"

class MinRV(OpRV):
    def __init__(self, d1, d2):
        super(MinRV, self).__init__([d1, d2], sym = None)
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        #return "min(#{0}, #{1})".format(id(self.d1), id(self.d2))
        return "min({0},{1})".format(self.d1, self.d2)
    def getName(self):
        return "min({0}, {1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "min"

class MaxRV(OpRV):
    def __init__(self, d1, d2):
        super(MaxRV, self).__init__([d1, d2], sym = None)
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        return "max(#{0}, #{1})".format(id(self.d1), id(self.d2))
    def getName(self):
        return "max({0}, {1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "max"

_builtin_min = min
def min(*args):
    if len(args) != 2:
        return _builtin_min(*args)
    d1 = args[0]
    d2 = args[1]
    if isinstance(d1, RV) and isinstance(d2, RV):
        return MinRV(d1, d2)
    elif isinstance(d1, RV) or isinstance(d2, RV):
        raise NotImplemented()
    else:
        return _builtin_min(*args)
_builtin_max = max
def max(*args):
    if len(args) != 2:
        return _builtin_max(*args)
    d1 = args[0]
    d2 = args[1]
    if isinstance(d1, RV) and isinstance(d2, RV):
        return MaxRV(d1, d2)
    elif isinstance(d1, RV) or isinstance(d2, RV):
        raise NotImplemented()
    else:
        return _builtin_max(*args)
if __name__ == "__main__":
    from pylab import *
    from pacal.distr import Distr
    x = RV(sym="x")
    y = RV(sym="y")
    z = RV(sym="z")
    u = x + y
    print ">>", u.getParentsAll()
    u.setSym("u")
    print u.getEquations()
    v = u + z 
    v.setSym("v")
    
    print v.getEquations()
    
    print u.getSym()
    print v.getSym()
    
    
#    print d;
#    print e;
#    print d.getEquations();
#    print d.getSym(), "=", d.getSymname();
#    x=RV(sym="x", a=1, b=3)
#    y=RV(sym="y")
#    z=RV(sym="z")
#    u=x+z/x
#    print u.getSym()
#    print u.isFree()    
#    print u.getParentsDep()
#    print u.getParentsFree()
#    print u.getParentsAll()
#    print u.a, u.b
#    print e.getEquations()
#    u.clear()
#    print e.getEquations()
#    
