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
from pacal.utils import lgamma


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
#   
#   #!-------------------
#   #! Section 9.1.3
#   #!-------------------
#   #! Harmonic mean of uniforms
#   for n in [2, 3, 7]:
#       d = 1 / UniformDistr(0, 1)
#       for i in xrange(n-1):
#           d += 1 / UniformDistr(0, 1)
#       d = n / d
#       figure()
#       demo_distr(d)
#   
#   
#   #!-------------------
#   #! Section 9.2
#   #!-------------------
#   #! tested elsewhere
#   
#   #!-------------------
#   #! Section 9.6
#   #!-------------------
#   from numpy import pi, sin, cos
#   for n1, n2, theta in [(5, 8, pi/4),
#                         #(5, 8, pi/2), does not work because of multiplication by ~0
#                         (25, 80, pi/8),
#                         (1, 1, pi/3),
#                         (2, 1, pi/3),
#                         ]:
#       d = StudentTDistr(n1) * sin(theta) + StudentTDistr(n2) * cos(theta)
#       figure()
#       demo_distr(d, xmin=-10, xmax=10)
#   
#   
#   #!-------------------
#   #! Section 9.9
#   #!-------------------
#   #! TODO: Bessel function distributions
#   
#   #! Corollary 9.9.1b
#   for n in [2, 3, 6]:
#       s = GammaDistr()
#       for i in xrange(n-1):
#           s += GammaDistr()
#       s /= n
#       figure() 
#       demo_distr(s, theoretical = GammaDistr(2*n, 2.0/n))
#   
#   
#   #! Theorem 9.9.2
def nonc_chi2(n, d):
    t = NormalDistr(sqrt(d))**2
    nc = t
    if n > 1:
        nc += ChiSquareDistr(n-1)
    #for i in xrange(n-1):
    #    nc += NormalDistr()**2
    return nc
#   for noncs in [[(2, 0), (1, 0)],
#                 [(1, 1.5), (1, 0.5)],
#                 [(3, 2.5), (2, 0.5)],
#                 ]:
#       nsum = noncs[0][0]
#       noncsum = noncs[0][1]
#       s = nonc_chi2(*noncs[0])
#       for n, d in noncs[1:]:
#           nsum += n
#           noncsum += d
#           s += nonc_chi2(n, d)
#       figure()
#       demo_distr(s, xmax=15, theoretical = nonc_chi2(nsum, noncsum))
#   
#   
#   #! Section 9.9.7
#   for n1, n2, xm in [(1, 1, 10),
#                  (2, 1, 10),
#                  (3, 5, 10),
#                  (13, 20, None),
#                  (130, 50, None),
#                  ]:
#       d = ChiSquareDistr(n1) * ChiSquareDistr(n2)
#       figure()
#       if xm is not None:
#           demo_distr(d, xmax=xm)
#       else:
#           demo_distr(d)
#           
#   # check this: first version of the nonc_chi2 gives better results for
#   # multiplication but worse for division!!!!
#   for n1, d1, n2, d2 in [(1, 0.4, 1, 2.1),
#                          (2, 1.5, 1, 0.7),
#                          (3, 0.15, 5, 4.1),
#                          ]:
#       d = nonc_chi2(n1, d1) * nonc_chi2(n2, d2)
#       figure()
#       demo_distr(d, xmax = 10)
#       d = nonc_chi2(n1, d1) / nonc_chi2(n2, d2)
#       figure()
#       demo_distr(d, xmax = 10)
#       #figure()
#       #d.get_piecewise_pdf().plot_tails()
#   
#   def quot_chi_theor(n1, n2, x):
#       n1 = float(n1)
#       n2 = float(n2)
#       t1 = n1/2
#       t2 = n2/2
#       ly =  lgamma(t1+t2) + t1*log(t1/4) + t2*log(t2/4) + (2*t1-1)*log(x)
#       ly -= lgamma(t1) + lgamma(t2) + (t1+t2)*log(t1/4*x*x+t2/4)
#       return 2*exp(ly)
#   for n1, n2 in [(1, 1),
#                  (2, 1),
#                  (3, 5),
#                  (13, 20),
#                  (130, 50),
#                  ]:
#       d = sqrt(ChiSquareDistr(n1)) * sqrt(ChiSquareDistr(n2))
#       figure()
#       demo_distr(d)
#       d2 = sqrt(ChiSquareDistr(n1)) / sqrt(ChiSquareDistr(n2))
#       figure()
#       demo_distr(d2)
#       # theoretical formula in Springer seems wrong:
#       #demo_distr(d2, theoretical = partial(quot_chi_theor, n1, n2))
#   
#   for n1, d1, n2, d2 in [(1, 0.4, 1, 2.1),
#                          (2, 1.5, 1, 0.7),
#                          (3, 0.15, 5, 4.1),
#                          ]:
#       nc1 = nonc_chi2(n1, d1)
#       nc2 = nonc_chi2(n2, d2)
#       d = nc1 * nc2
#       figure()
#       demo_distr(d, xmax = 10)
#       d = nc1 / nc2
#       figure()
#       demo_distr(d, xmax = 10)
#   
#   #! folded normal distribution
#   def theor_quot_folded_normal(sigma1, sigma2, x):
#       return 2*sigma1*sigma2 / numpy.pi / (sigma1**2 + x**2*sigma2**2)
#   fn = abs(NormalDistr())
#   figure()
#   demo_distr(fn * fn)
#   figure()
#   demo_distr(fn / fn, theoretical = partial(theor_quot_folded_normal, 1, 1))
#   
#   
#   #!-------------------
#   #! Section 9.10
#   #!-------------------
#   #! Linear combination of truncated exponential distributions
#   def truncExp(alpha, theta):
#       return CondLtDistr(ExponentialDistr(alpha), theta)
#   
#   demo_distr(truncExp(1, 10))
#   te1 = truncExp(1, 10)
#   demo_distr(te1 + te1 + te1)
#   demo_distr(te1 + 10*truncExp(2, 5))
#   
#   #!-------------------------
#   #! Section 9.11
#   #! Generalized F variables
#   #!-------------------------
#   
#   def gen_f(p, m, a, h, x):
#       h = float(h)
#       logk = p/h*log(a)+lgamma(m)-lgamma(p/h)-lgamma(m-p/h)
#       k = h * exp(logk)
#       return k * x**(p-1) / (1 + a*x**h)**m
#   f = FunDistr(partial(gen_f, 1, 1, 1, 2), [0,1,Inf])
#   
#   demo_distr(f, theoretical = abs(CauchyDistr()))
#   
#   for genfs in [[(1, 1, 1, 2), (1, 1, 1, 2)],
#                 [(1, 1, 1, 2), (3, 2, 1, 2), (1, 7, 0.0, 3)]
#                 ]:
#       pr = OneDistr()
#       for gf in genfs:
#           f = FunDistr(partial(gen_f, 1, 1, 1, 2), [0,1,Inf])
#           pr *= f
#       figure()
#       pr.plot()
#       pr.summary()

#!-------------------------
#! Section 9.12
#!-------------------------

for ps in [[1,2,3],
           [6,8,10,12]]:
    pqs = zip(ps[:-1], ps[1:])
    pr = OneDistr()
    for p, q in pqs:
        pr *= BetaDistr(p, q-p)
    figure()
    demo_distr(pr, theoretical = BetaDistr(ps[0], ps[-1]-ps[0]))

show()
