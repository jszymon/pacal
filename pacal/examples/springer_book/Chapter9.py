#!
#!--------------------------------------
#! CHAPTER 9 - STATISTICAL APPLICATIONS
#!--------------------------------------
#!
from functools import partial
import numpy

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr


#!-------------------
#! Section 9.1
#!-------------------

#! Example 9.1.1
#! Implemented elsewhere

#   #! Example 9.1.2
#   def theor_sum_exp(a1, a2, a3, x):
#       t1 = exp(-3*a1*x) / (a2-a1) / (a3-a1)
#       t2 = exp(-3*a2*x) / (a1-a2) / (a3-a2)
#       t3 = exp(-3*a3*x) / (a1-a3) / (a2-a3)
#       return 3*a1*a2*a3 * (t1+t2+t3)
#   for a1, a2, a3 in [(1.0,2.0,3.0),
#                      (1,0.01,100.0),]:
#       figure()
#       d = (ExponentialDistr(a1) + ExponentialDistr(a2) + ExponentialDistr(a3))/3
#       demo_distr(d, theoretical = partial(theor_sum_exp, a1, a2, a3))

#! Section 9.1.2
#! the L1 statistic for variances
#! Question: are the numerator and denominator independent?
ns = [3, 5, 1] # sample sizes
N = sum(ns)
num = ChiSquareDistr(ns[0] - 1)
for n in ns[1:]:
    num *= ChiSquareDistr(n - 1)
num **= (1.0 / N)
den = ChiSquareDistr(N - len(ns))
L1 = num / den
demo_distr(L1)


show()
