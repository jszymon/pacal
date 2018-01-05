#!
#!--------------------------------------
#! CHAPTER 9 - STATISTICAL APPLICATIONS
#!--------------------------------------
#!
from __future__ import print_function

from functools import partial
import numpy

from pylab import figure, show

# Those need to be defined before PaCAL creates a process pool for
# picklability:
def example_f1(x):
    return sqrt(2)/numpy.pi / (1+x**4)
def example_f2(x):
    return 1.5*x*x


from pacal import *
from pacal.distr import demo_distr
from pacal.utils import lgamma

import time

# helper functions used below
def gen_f(p, m, a, h, x):
    h = float(h)
    logk = p/h*log(a)+lgamma(m)-lgamma(p/h)-lgamma(m-p/h)
    k = h * exp(logk)
    return k * x**(p-1) / (1 + a*x**h)**m
grcache = {}
def gr1(m, p, p0, r, x):
    ly = lgamma(float(p+r-1)/m)-lgamma(float(p0+r-1)/m)-lgamma(float(p-p0)/m) + (p0+r-1)*log(x)
    y = m * exp(ly)*(1.0-x**m) **(float(p-p0)/m - 1)
    return y
def gr(m, p, p0, r, x):
    """Need to scale the distributions.  Examples in Springer/Kotlarski seem not normalized"""
    if r in grcache:
        scale = grcache[r]
    else:
        f = FunDistr(partial(gr1, m, p, p0, r), [0,1])
        scale = f.cdf(1)
        grcache[r] = scale
    y = gr1(m, p, p0, r, x)
    return y / scale

if __name__ == "__main__":
    #!-------------------
    #! Section 9.1
    #!-------------------
    
    #! Example 9.1.1
    #! Implemented elsewhere
    
    #! Example 9.1.2
    t0 = time.time()
    def theor_sum_exp(a1, a2, a3, x):
        t1 = exp(-3*a1*x) / (a2-a1) / (a3-a1)
        t2 = exp(-3*a2*x) / (a1-a2) / (a3-a2)
        t3 = exp(-3*a3*x) / (a1-a3) / (a2-a3)
        return 3*a1*a2*a3 * (t1+t2+t3)
    for a1, a2, a3 in [(1.0,2.0,3.0),
                       (1,0.01,100.0),]:
        figure()
        d = (ExponentialDistr(a1) + ExponentialDistr(a2) + ExponentialDistr(a3))/3
        demo_distr(d, theoretical = partial(theor_sum_exp, a1, a2, a3))
    
    #! Section 9.1.2
    #! the L1 statistic for variances
    #! Question: are the numerator and denominator independent?
    for ns in [[3, 5, 2], # sample sizes
               [4, 5, 10, 7, 3]
               ]:
        print("sample sizes:", ns)
        N = sum(ns)
        num = ChiSquareDistr(ns[0] - 1)
        for n in ns[1:]:
            num *= ChiSquareDistr(n - 1)
        #num.summary()
        num **= (1.0 / N)
        den = ChiSquareDistr(N - len(ns)) / N
        L1 = num / den
        figure()
        demo_distr(L1, xmax=10)
    
    #! Example 9.1.2
    #! Geometric mean of uniforms
    def theor_geom_unif(n, x):
        nf = 1
        for i in range(2, n):
            nf *= i
        return float(n) / nf * x**(n-1) * log(x**(-n))**(n-1)
    for n in [3, 7]:
        d = UniformDistr(0, 1)
        for i in range(n-1):
            d *= UniformDistr(0, 1)
        d **= (1.0 / n)
        d2 = log(UniformDistr(0, 1))
        for i in range(n-1):
            d2 += log(UniformDistr(0, 1))
        d2 /= n
        d2 = exp(d2)
        figure()
        demo_distr(d, theoretical = partial(theor_geom_unif, n))
        figure()
        demo_distr(d2, theoretical = partial(theor_geom_unif, n))
    
    #!-------------------
    #! Section 9.1.3
    #!-------------------
    #! Harmonic mean of uniforms
    for n in [2, 3, 7]:
        d = 1 / UniformDistr(0, 1)
        for i in range(n-1):
            d += 1 / UniformDistr(0, 1)
        d = n / d
        figure()
        demo_distr(d)
    
    
    #!-------------------
    #! Section 9.2
    #!-------------------
    #! tested elsewhere
    
    #!-------------------
    #! Section 9.6
    #!-------------------
    from numpy import pi, sin, cos
    for n1, n2, theta in [(5, 8, pi/4),
                          #(5, 8, pi/2), does not work because of multiplication by ~0
                          (25, 80, pi/8),
                          (1, 1, pi/3),
                          (2, 1, pi/3),
                          ]:
        d = StudentTDistr(n1) * sin(theta) + StudentTDistr(n2) * cos(theta)
        figure()
        demo_distr(d, xmin=-10, xmax=10)
    
    
    #!-------------------
    #! Section 9.9
    #!-------------------
    #! TODO: Bessel function distributions
    
    #! Corollary 9.9.1b
    for n in [2, 3, 6]:
        s = GammaDistr()
        for i in range(n-1):
            s += GammaDistr()
        s /= n
        figure() 
        demo_distr(s, theoretical = GammaDistr(2*n, 2.0/n))
    
    
    #! Theorem 9.9.2
    for noncs in [[(2, 0), (1, 0)],
                  [(1, 1.5), (1, 0.5)],
                  [(3, 2.5), (2, 0.5)],
                  ]:
        nsum = noncs[0][0]
        noncsum = noncs[0][1]
        s = NoncentralChiSquareDistr(*noncs[0])
        for n, d in noncs[1:]:
            nsum += n
            noncsum += d
            s += NoncentralChiSquareDistr(n, d)
        figure()
        demo_distr(s, xmax=15, theoretical = NoncentralChiSquareDistr(nsum, noncsum))
    
    
    #! Section 9.9.7
    for n1, n2, xm in [(1, 1, 10),
                   (2, 1, 10),
                   (3, 5, 10),
                   (13, 20, None),
                   (130, 50, None),
                   ]:
        d = ChiSquareDistr(n1) * ChiSquareDistr(n2)
        figure()
        if xm is not None:
            demo_distr(d, xmax=xm)
        else:
            demo_distr(d)
            
    for n1, d1, n2, d2 in [(1, 0.4, 1, 2.1),
                           (2, 1.5, 1, 0.7),
                           (3, 0.15, 5, 4.1),
                           ]:
        d = NoncentralChiSquareDistr(n1, d1) * NoncentralChiSquareDistr(n2, d2)
        figure()
        demo_distr(d, xmax = 10)
        d = NoncentralChiSquareDistr(n1, d1) / NoncentralChiSquareDistr(n2, d2)
        figure()
        demo_distr(d, xmax = 10)
        #figure()
        #d.get_piecewise_pdf().plot_tails()
    
    def quot_chi_theor(n1, n2, x):
        n1 = float(n1)
        n2 = float(n2)
        t1 = n1/2
        t2 = n2/2
        ly =  lgamma(t1+t2) + t1*log(t1/4) + t2*log(t2/4) + (2*t1-1)*log(x)
        ly -= lgamma(t1) + lgamma(t2) + (t1+t2)*log(t1/4*x*x+t2/4)
        return 2*exp(ly)
    for n1, n2 in [(1, 1),
                   (2, 1),
                   (3, 5),
                   (13, 20),
                   (130, 50),
                   ]:
        d = sqrt(ChiSquareDistr(n1)) * sqrt(ChiSquareDistr(n2))
        figure()
        demo_distr(d)
        d2 = sqrt(ChiSquareDistr(n1)) / sqrt(ChiSquareDistr(n2))
        figure()
        demo_distr(d2)
        # theoretical formula in Springer seems wrong:
        #demo_distr(d2, theoretical = partial(quot_chi_theor, n1, n2))
    
    for n1, d1, n2, d2 in [(1, 0.4, 1, 2.1),
                           (2, 1.5, 1, 0.7),
                           (3, 0.15, 5, 4.1),
                           ]:
        nc1 = NoncentralChiSquareDistr(n1, d1)
        nc2 = NoncentralChiSquareDistr(n2, d2)
        d = nc1 * nc2
        figure()
        demo_distr(d, xmax = 10)
        d = nc1 / nc2
        figure()
        demo_distr(d, xmax = 10)
    
    #! folded normal distribution
    def theor_quot_folded_normal(sigma1, sigma2, x):
        return 2*sigma1*sigma2 / numpy.pi / (sigma1**2 + x**2*sigma2**2)
    fn = abs(NormalDistr())
    figure()
    params.general.warn_on_dependent = False
    demo_distr(fn * fn)
    figure()
    demo_distr(fn / fn, theoretical = partial(theor_quot_folded_normal, 1, 1))
    params.general.warn_on_dependent = True
    
    
    #!-------------------
    #! Section 9.10
    #!-------------------
    #! Linear combination of truncated exponential distributions
    def truncExp(alpha, theta):
        return CondLtDistr(ExponentialDistr(alpha), theta)
    
    figure()
    demo_distr(truncExp(1, 10), err_plot = False)
    te1 = truncExp(1, 10)
    figure()
    demo_distr(iid_sum(te1, 3))
    figure()
    demo_distr(te1 + 10*truncExp(2, 5))
    
    #!---------------------------------------
    #! Section 9.11: Generalized F variables
    #!---------------------------------------
    
    f = FunDistr(partial(gen_f, 1, 1, 1, 2), [0,1,Inf])
    
    figure()
    demo_distr(f, theoretical = abs(CauchyDistr()))
    
    for genfs in [[(1, 1, 1, 2), (1, 1, 1, 2)],
                  [(1, 1, 1, 2), (3, 2, 1, 2), (1, 7, 0.0, 3)]
                  ]:
        pr = OneDistr()
        for gf in genfs:
            f = FunDistr(partial(gen_f, 1, 1, 1, 2), [0,1,Inf])
            pr *= f
        figure()
        pr.plot()
        pr.summary()
    
    #!-------------------------
    #! Section 9.13
    #!-------------------------
    
    for ps in [[1,2,3],
               [6,8,10,12]]:
        pqs = list(zip(ps[:-1], ps[1:]))
        pr = OneDistr()
        for p, q in pqs:
            pr *= BetaDistr(p, q-p)
        figure()
        demo_distr(pr, theoretical = BetaDistr(ps[0], ps[-1]-ps[0]))
    
    #! Exaple 9.13.1 (following Kotlarski)
    p0 = 2; p = 5
    m = 3        
    
    pr = OneDistr()
    for i in range(m):
        f = FunDistr(partial(gr, m, p, p0, i), [0,1])
        pr *= f
    figure()
    demo_distr(pr, theoretical = BetaDistr(p0, p-p0))
    
    
    #!-------------------------
    #! Section 9.14
    #!-------------------------
    
    f = FunDistr(example_f1 , [-Inf, -1, 0, 1, Inf])
    figure()
    demo_distr(f, err_plot = False)
    figure()
    params.general.warn_on_dependent = False
    demo_distr(f/f, theoretical = CauchyDistr())
    params.general.warn_on_dependent = True
    
    
    
    
    #! Exercise 9.1
    for l, n in [[1, 2],
                 [1, 5],
                 ]:
        e = ExponentialDistr(l)
        s = ZeroDistr()
        for i in range(n):
            s += e
        s /= n
        figure()
        demo_distr(s, theoretical = GammaDistr(n, 1)/n)
    
    #! Exercise 9.2
    params.general.warn_on_dependent = False
    f = FunDistr(example_f2, [-1,1])
    m = (f + f) / 2
    figure()
    demo_distr(m)
    #! Exercise 9.3
    m2 = (m + m) / 2
    figure()
    demo_distr(m2)
    params.general.warn_on_dependent = True
    
    #! Exercise 9.4
    for k, n in [[2,2]]:
        s = ZeroDistr()
        for i in range(n):
            s += GammaDistr(k)
        figure()
        demo_distr(s, theoretical = GammaDistr(k*n))
    
    #! Exercise 9.5: see central_limit_demo.py
    
    #! Exercise 9.6
    maxwell = sqrt(NormalDistr()**2 + NormalDistr()**2 + NormalDistr()**2)
    for n in [2, 3, 5]:
        s = iid_average(maxwell, n)
        figure()
        demo_distr(s)
    
    #! Exercise 9.7
    rayleigh = sqrt(NormalDistr()**2 + NormalDistr()**2)
    for n in [2, 3, 5]:
        s = iid_average(rayleigh, n)
        figure()
        demo_distr(s)
    
    #! Exercise 9.8: see Exercise 9.4
    
    #! Exercise 9.11
    for n, p in [[2, 5]]:
        pr = OneDistr()
        for i in range(n):
            pr *= GammaDistr(p+float(i)/n, 1)
        gm = pr ** (1.0/n)
        figure()
        demo_distr(gm, theoretical = GammaDistr(n*p, 1) / n)
    
    #! Exercise 9.12
    figure()
    demo_distr((BetaDistr(5, 2) * BetaDistr(6, 2) * BetaDistr(6, 3))**(1.0/3), xmin = 0, xmax = 1)
    
    #! Exercise 9.13: see singularities.py
    
    #! Exercise 9.14: see Exercise 4.2
    for u1, u2 in [[-1,1],
                   [4.5,0.5],
                   [0.5,0.5],
                   ]:
        p = UniformDistr(u1-0.5, u1+0.5) * UniformDistr(u2-0.5, u2+0.5)
        figure()
        demo_distr(p)
    
    #! Exercise 9.15:
    for a, b, c in [[1,2,3]]:
        pr = BetaDistr(a, b) * BetaDistr(a+b, c)
        figure()
        demo_distr(pr, theoretical = BetaDistr(a, b+c))
    
    #! Exercise 9.16: see Section 9.13
    
    #! Exercise 9.17
    a1 = 3
    bs = [2,3]
    p = len(bs)
    As = [a1]
    for i in range(p-1):
        As.append(As[i] + bs[i])
    pr = OneDistr()
    for i in range(p):
        pr *= 1/(1+GammaDistr(bs[i], 1) / GammaDistr(As[i], 1))
    figure()
    demo_distr(pr, theoretical = BetaDistr(a1, sum(bs)))
    
    #! Exercises 9.21-9.27
    for a1, b1, a2, b2 in [[2, 0, 0.5, -1], #FIX: !!!! discontinuity at 0!!!! singularity in derivative not detected?
                           [1, 0, 1, 1],
                           [4, 0, 4, 0],
                           ]:
        tr = UniformDistr() + UniformDistr()
        tr1 = b1 + tr/sqrt(a1)
        tr2 = b2 + tr/sqrt(a2)
        figure()
        params.general.warn_on_dependent = False
        pr = tr1 * tr2
        params.general.warn_on_dependent = True
        demo_distr(pr)
    
    #! Exercise 9.28: see Section 9.14
    
    #! Exercise 9.29: see Section 9.1.2
    
    #! Exercise 9.34
    figure()
    demo_distr(StudentTDistr(5) - StudentTDistr(5))
    
    
    #! Exercises 9.35, 9.36
    n = 5
    v = abs(NormalDistr()) * abs(NormalDistr()) / ChiSquareDistr(n) * n
    figure()
    demo_distr(v, xmax = 10, ymax = 2)
    
    #! Exercise 9.37
    for n1, n2 in [[1, 1],
                   [2, 3],
                   [5, 10],
                   [5, 100],
                   ]:
        c1 = ChiSquareDistr(n1)
        c2 = ChiSquareDistr(n2)
        v = 1/(1+c2/c1)
        figure()
        demo_distr(v, theoretical = BetaDistr(n1/2.0, n2/2.0))
    
    #! Exercise 9.37
    for p in [1, 3, 5, 100]:
        figure()
        demo_distr(GammaDistr(p, 1) - GammaDistr(p, 1))
    print(time.time() - t0)
    show()
