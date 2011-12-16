"""Utilities for sympy.

Mainly work around incompatibilities between versions."""

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
