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
#   
#   #! Section 9.1.2
#   #! the L1 statistic for variances
#   #! Question: are the numerator and denominator independent?
#   for ns in [[3, 5, 2], # sample sizes
#              [4, 5, 10, 7, 3]
#              ]:
#       print "sample sizes:", ns
#       N = sum(ns)
#       num = ChiSquareDistr(ns[0] - 1)
#       for n in ns[1:]:
#           num *= ChiSquareDistr(n - 1)
#       #num.summary()
#       num **= (1.0 / N)
#       den = ChiSquareDistr(N - len(ns)) / N
#       L1 = num / den
#       figure()
#       demo_distr(L1, xmax=10)
#   
#   #! Example 9.1.2
#   #! Geometric mean of uniforms
#   def theor_geom_unif(n, x):
#       nf = 1
#       for i in xrange(2, n):
#           nf *= i
#       return float(n) / nf * x**(n-1) * log(x**(-n))**(n-1)
#   for n in [3, 7]:
#       d = UniformDistr(0, 1)
#       for i in xrange(n-1):
#           d *= UniformDistr(0, 1)
#       d **= (1.0 / n)
#       d2 = log(UniformDistr(0, 1))
#       for i in xrange(n-1):
#           d2 += log(UniformDistr(0, 1))
#       d2 /= n
#       d2 = exp(d2)
#       figure()
#       demo_distr(d, theoretical = partial(theor_geom_unif, n))
#       figure()
#       demo_distr(d2, theoretical = partial(theor_geom_unif, n))

#! Section 9.1.3
#! Harmonic mean of uniforms
for n in [2, 3, 7]:
    d = 1 / UniformDistr(0, 1)
    for i in xrange(n-1):
        d += 1 / UniformDistr(0, 1)
    d = n / d
    figure()
    demo_distr(d)


#!-------------------
#! Section 9.2
#!-------------------
#! tested elsewhere



show()
