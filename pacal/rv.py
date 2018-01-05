"""Random Variable class"""

import numbers
from os import getpid

import numpy
from numpy import unique
from numpy import add, subtract, divide, prod, multiply

import sympy
from sympy import var

from . import params
from .sympy_utils import sympy_min, sympy_max, sympy_abs

class RV(object):
    def __init__(self, parents = [], sym = None, a=0.0, b=1.0):
        self.parents = parents
        self.a = a
        self.b = b
        if sym is not None:         # ====set symbolic name of r.v
            if isinstance(sym, str):
                self.sym = sympy.Symbol(sym)
            else:
                self.sym = sym      # user defined symbolic name of r.v.
        else:
            self.sym = sympy.Symbol("X{0}".format(self.id())) # default symbolic name of r.v.
        if self.sym.is_Atom:
            #print "atom: ", self.sym
            self.symname = self.sym
        else:
            #print "complex: ", self.sym
            self.symname = sympy.Symbol("X{0}".format(self.id()))
    def id(self):
        """Return an object id which can survive pickling in parallel
        mode."""
        if params.general.parallel:
            if not hasattr(self, "_id"):
                self._id = str(getpid()) + "_" + str(id(self))
            rvid = self._id
        else:
            rvid = id(self)
        return rvid
    def __str__(self):
        return "RV(" + str(self.sym) + ")"
    def __repr__(self):
        return str(self)

    def getAncestorIDs(self, anc = None):
        """Get ID's of all ancestors"""
        if anc is None:
            anc = set()
        for p in self.parents:
            if p.id() not in anc:
                anc.update(p.getAncestorIDs(anc))
        anc.add(self.id())
        return anc

    def range(self):
        return self.a, self.b

    def getName(self):
        """return, string representation of RV"""
        return str(self)
    #def getExpr
    def getSym(self):
        """return, symbolic representation of RV"""
        return self.sym

    def setSym(self, sym, make_free = False):
        """Set the symbolic name of RV."""

        if isinstance(sym, str):
            self.symname = sympy.Symbol(sym)
        else:
            self.symname = sym
    def make_free(self):
        """Removes the equation of the variable and its parents.  The variable
        becomes free."""
        self.sym = self.symname
        self.parents = []
    #def getSymbol
    def getSymname(self):
        return self.symname

    def getEquations(self, node=None, l=None, r=None):
        if l is None:
            l = []
        if r is None:
            r = []
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
    def getParentsAll(self):
        l = set([self])
        for p in self.parents:
            l.update(p.getParentsDep())
        return l
    def getParentsFree(self):
        l = set()
        if len(self.parents) == 0:
            l.add(self)
        for p in self.parents:
            l.update(p.getParentsFree())
        return l
    def getParentsDep(self):
        l = set()
        if len(self.parents) > 0:
            l.add(self)
        for p in self.parents:
            l.update(p.getParentsDep())
        return l

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
        return NotImplemented
    def __radd__(self, d):
        """Overload sum with real number: distribution of X+r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, shift = d)
        return NotImplemented
    def __sub__(self, d):
        """Overload subtraction: distribution of X-Y."""
        if isinstance(d, RV):
            return SubRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, shift = -d)
        return NotImplemented
    def __rsub__(self, d):
        """Overload subtraction with real number: distribution of X-r."""
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, scale = -1, shift = d)
        return NotImplemented
    def __mul__(self, d):
        """Overload multiplication: distribution of X*Y."""
        if isinstance(d, RV):
            return MulRV(self, d)
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledRV(self, scale = d)
        return NotImplemented
    def __rmul__(self, d):
        """Overload multiplication by real number: distribution of X*r."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            else:
                return ShiftedScaledRV(self, scale = d)
        return NotImplemented
    def __truediv__(self, d):
        """Overload division: distribution of X/r."""
        if isinstance(d, RV):
            return DivRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, scale = 1.0 / d)
        return NotImplemented
    def __div__(self, d):
        """Python2 version."""
        if isinstance(d, RV):
            return DivRV(self, d)
        if isinstance(d, numbers.Real):
            return ShiftedScaledRV(self, scale = 1.0 / d)
        return NotImplemented
    def __rtruediv__(self, d):
        """Overload division by real number: distribution of r/X."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            return d * InvRV(self)
        return NotImplemented
    def __rdiv__(self, d):
        """Python2 version."""
        if isinstance(d, numbers.Real):
            if d == 0:
                return 0
            d = float(d)
            return d * InvRV(self)
        return NotImplemented
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
        return NotImplemented
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
        return NotImplemented


def _wrapped_name(d, incl_classes = None):
    """Return name of d wrapped in parentheses if necessary"""
    d_name = d.getName()
    if incl_classes is not None:
        if isinstance(d, tuple(incl_classes)):
            d_name = "(" + d_name + ")"
    elif isinstance(d, OpRV) and not isinstance(d, (FuncRV, SquareRV)):
        d_name = "(" + d_name + ")"
    return d_name


class OpRV(RV):
    """Base class for operations on RVs."""
    def __str__(self):
        if len(self.parents)==2:
            if isinstance(self.getSym(), sympy.Add):
                op = "+"
            if isinstance(self.getSym(), sympy.Mul):
                op = "*"
            else:
                op = str(self.getSym().__class__)
            return "({0}{2}{1})".format(self.parents[0], self.parents[1], op)
class FuncRV(OpRV):
    """Function of random variable"""
    def __init__(self, d, fname = "f", sym = None):
        super(FuncRV, self).__init__([d], sym = sym)
        self.d = d
        self.fname = fname
    def __str__(self):
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
        super(ShiftedScaledRV, self).__init__([d], sym = (d.getSymname() * self.scale + self.shift))
        a = d.a * self.scale + self.shift
        b = d.b * self.scale + self.shift
        self.a, self.b = min(a, b), max(a, b)
        self.d = d
    def __str__(self):
        if self.shift == 0 and self.scale == 1:
            return str(self.d.id())
        elif self.shift == 0:
            return "{0}*#{1}".format(self.scale, self.d.id())
        elif self.scale == 1:
            return "#{0}{1:+}".format(self.d.id(), self.shift)
        else:
            return "#{0}*{1}{2:+}".format(self.d.id(), self.scale, self.shift)
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

class ExpRV(FuncRV):
    """Exponent of a random variable"""
    def __init__(self, d):
        super(ExpRV, self).__init__(d, fname = "exp", sym = sympy.exp(d.getSymname()))
    def is_nonneg(self):
        return True
class LogRV(FuncRV):
    """Natural logarithm of a random variable"""
    def __init__(self, d):
        if not d.is_nonneg():
            raise ValueError("logarithm of a nonpositive distribution")
        super(LogRV, self).__init__(d, fname = "log", sym = sympy.log(d.getSymname()))

class AtanRV(FuncRV):
    """Arcus tangent of a random variable"""
    def __init__(self, d):
        super(AtanRV, self).__init__(d, fname ="atan", sym = sympy.atan(d.getSymname()))

class TanhRV(FuncRV):
    """Hyperbolic tangent of a random variable"""
    def __init__(self, d):
        super(TanhRV, self).__init__(d, fname ="tanh", sym = sympy.tanh(d.getSymname()))

class InvRV(OpRV):
    """Inverse of random variable."""
    def __init__(self, d):
        super(InvRV, self).__init__([d], sym = 1 / d.getSymname())
        self.d = d
    def __str__(self):
        return "1/#{0}".format(self.d.id())
    #def getName(self):
    #    return "1/{0}".format(self.d.getName())
    def getName(self):
        d_name = self.d.getName()
        if isinstance(self.d, OpRV) and not isinstance(self.d, FuncRV):
            d_name = "(" + d_name + ")"
        return "(1/{0})".format(d_name)

class PowRV(FuncRV):
    """Inverse of random variable."""
    def __init__(self, d, alpha = 1):
        super(PowRV, self).__init__([d], fname="pow", sym = d.getSymname()**alpha)
        self.d = d
        self.alpha = alpha
    def __str__(self):
        return "#{0}**{1}".format(self.d1.id(), self.alpha)
    def getName(self):
        return "{0}**{1}".format(_wrapped_name(self.d), self.alpha)

class AbsRV(OpRV):
    """Absolute value of a distribution."""
    def __init__(self, d):
        super(AbsRV, self).__init__([d], sym = sympy_abs(d.getSymame())) # TODO abs unpleasant in sympy
        self.d = d
    def __str__(self):
        return "|#{0}|".format(self.d.id())
    def getName(self):
        return "|{0}|".format(self.d.getName())

class SignRV(RV):
    def __init__(self, d):
        self.d = d
        super(SignRV, self).__init__(d, sym = sympy.sign(d.getSymname()))
    def __str__(self):
        return "sign({0})".format(self.d.id())
    def getName(self):
        return "sign({0})".format(self.d.getName())

class SquareRV(OpRV):
    """Injective function of random variable"""
    def __init__(self, d):
        super(SquareRV, self).__init__([d], sym = d.getSymname()**2)
        self.d = d
    def __str__(self):
        return "#{0}**2".format(self.d.id())
    def getName(self):
        return "sqr({0})".format(self.d.getName())


class SumRV(OpRV):
    """Sum of distributions."""
    def __init__(self, d1, d2):
        super(SumRV, self).__init__([d1, d2], sym = d1.getSymname() + d2.getSymname())
        #breaks = unique(add.outer(d1.range(), d2.range()))
        #self.a, self.b = min(breaks), max(breaks)
        a1, b1 = d1.range()
        a2, b2 = d2.range()
        self.a, self.b = a1+a2, b1+b2
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}+#{1}".format(self.d1.id(), self.d2.id())
    def getName(self):
        return "{0}+{1}".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "+"
class SubRV(OpRV):
    """Difference of distributions."""
    def __init__(self, d1, d2):
        super(SubRV, self).__init__([d1, d2], sym = d1.getSymname() - d2.getSymname())
        #breaks = unique(subtract.outer(d1.range(), d2.range()))
        #self.a, self.b = min(breaks), max(breaks)
        a1, b1 = d1.range()
        a2, b2 = d2.range()
        self.a, self.b = a1-a2, b1-b2
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}-#{1}".format(self.d1.id(), self.d2.id())
    def getName(self):
        n2 = _wrapped_name(self.d2, incl_classes = [SumRV])
        return "{0}-{1}".format(self.d1.getName(), n2)
    def getOperation(self):
        return "-"

class MulRV(OpRV):
    def __init__(self, d1, d2):
        super(MulRV, self).__init__([d1, d2], sym = d1.getSymname() * d2.getSymname())
        breaks = unique(multiply.outer(d1.range(), d2.range()))
        self.a, self.b = min(breaks), max(breaks)
        self.d1 = d1
        self.d2 = d2
    def __str__(self):
        return "#{0}*#{1}".format(self.d1.id(), self.d2.id())
    def getName(self):
        n1 = _wrapped_name(self.d1, incl_classes = [SumRV, SubRV, ShiftedScaledRV])
        n2 = _wrapped_name(self.d2, incl_classes = [SumRV, SubRV, ShiftedScaledRV])
        return "{0}*{1}".format(n1, n2)
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
    def __str__(self):
        return "#{0}/#{1}".format(self.d1.id(), self.d2.id())
    def getName(self):
        n1 = _wrapped_name(self.d1)
        n2 = _wrapped_name(self.d2)
        return "{0}/{1}".format(n1, n2)
    def getOperation(self):
        return "/"

class MinRV(OpRV):
    def __init__(self, d1, d2):
        super(MinRV, self).__init__([d1, d2], sym = sympy_min(d1.getSymname(), d2.getSymname()))
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        #return "min(#{0}, #{1})".format(self.d1.id(), self.d2.id())
        return "min({0},{1})".format(self.d1, self.d2)
    def getName(self):
        return "min({0}, {1})".format(self.d1.getName(), self.d2.getName())
    def getOperation(self):
        return "min"

class MaxRV(OpRV):
    def __init__(self, d1, d2):
        super(MaxRV, self).__init__([d1, d2], sym = sympy_max(d1.getSymname(), d2.getSymname()))
        self.d1 = d1
        self.d2 = d2
    def __str(self):
        return "max(#{0}, #{1})".format(self.d1.id(), self.d2.id())
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
        raise TypeError("unorderable types: {}() < {}()".format(type(d1).__name__, type(d2).__name__))
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
        raise TypeError("unorderable types: {}() < {}()".format(type(d1).__name__, type(d2).__name__))
    else:
        return _builtin_max(*args)


if __name__ == "__main__":
    x = RV(sym="x")
    y = RV(sym="y")
    z = RV(sym="z")
    u = x + y
    print(">>", u.getParentsAll())
    print(">>", u.getParentsFree())
    u.setSym("u")
    print(u.getEquations())
    v = u + z
    v.setSym("v")

    print(v.getEquations())

    print(u.getSym())
    print(v.getSym())


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
