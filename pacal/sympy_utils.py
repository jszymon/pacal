"""Utilities for sympy.

Mainly work around incompatibilities between versions."""

from functools import partial
from numpy import isfinite
import numpy as np
import sympy
import sympy.printing.lambdarepr

from . import params

# sympy defines max, min and abs differently in different versions
try:
    sympy_max = sympy.Max
except:
    sympy_max = sympy.max_
try:
    sympy_min = sympy.Min
except:
    sympy_min = sympy.min_
try:
    sympy_abs = sympy.Abs
except:
    sympy_abs = sympy.abs

# testing if sympy object
try:
    _sympy_base_class = sympy.Expr
except:
    _sympy_base_class = sympy.Basic
def is_sympy(x):
    return isinstance(x, _sympy_base_class)

# converting to sympy taking into account infinities
def sympify(x):
    if x == -sympy.oo:
        sx = -sympy.oo
    elif x == sympy.oo:
        sx = sympy.oo
    else:
        sx = sympy.sympify(x)
    return sx

# incompatibilities in equation solving semantics
# our own improvements
_eq_cache = {}
def eq_solve(lhs, rhs, x):
    key = (lhs, rhs, x)
    rhs = sympify(rhs)
    if key in _eq_cache:
        solutions = _eq_cache[key]
    else:
        if rhs == sympy.oo or rhs == -sympy.oo:
            tmp_rhs = sympy.Symbol("__tmp_rhs_" + str(id(rhs)))
            solutions = sympy.solve(lhs - tmp_rhs, x)
            solutions = [sympy.simplify(sympy.limit(s, tmp_rhs, rhs)) for s in solutions]
        else:
            solutions = sympy.solve(lhs - rhs, x)
        _eq_cache[key] = solutions
    return solutions

_numpy_funcs = [("sqrt", np.sqrt), ("log", np.log), ("exp", np.exp)]
def _my_lambdify_helper(expr_str, argnames, *argvals):
    #print expr_str, argnames, argvals, dict(_numpy_funcs + zip(argnames, argvals))
    return eval(expr_str, globals(), dict(_numpy_funcs + list(zip(argnames, argvals))))
def my_lambdify(args, expr, modules=None, printer=None, use_imps=True):
    """Lambdify sympy expressions without using lambda functions."""
    if params.general.parallel:
        argnames = [str(arg) for arg in args]
        l = partial(_my_lambdify_helper, sympy.printing.lambdarepr.lambdarepr(expr), argnames)
    else:
        l = sympy.lambdify(args, expr, modules, printer, use_imps)
    return l
