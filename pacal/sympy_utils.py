"""Utilities for sympy.

Mainly work around incompatibilities between versions."""

from numpy import isfinite
import sympy


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

# incompatibilities in equation solving semantics
# our own improvements
_eq_cache = {}
def eq_solve(lhs, rhs, x):
    print lhs, rhs, x
    key = (lhs, rhs, x)
    if key in _eq_cache:
        solutions = _eq_cache[key]
    else:
        if (is_sympy(rhs) and sympy_abs(rhs) == sympy.oo) or (not is_sympy(rhs) and not isfinite(rhs)):
            tmp_rhs = sympy.Symbol("__tmp_rhs_" + str(id(rhs)))
            solutions = sympy.solve(lhs - tmp_rhs, x)
            solutions = [sympy.simplify(sympy.limit(s, tmp_rhs, sympy.S(rhs))) for s in solutions]
        else:
            solutions = sympy.solve(lhs - rhs, x)
        _eq_cache[key] = solutions
    print solutions
    print
    return solutions
