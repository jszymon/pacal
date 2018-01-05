#!
#!----------------------
#! CHAPTER 4 - PRODUCTS
#!----------------------
#!
from __future__ import print_function

from functools import partial
import numpy

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr
import time


def funEx4_20(m, a, b, x):
    return (m + 1) * x / (b ** (m + 1) - a ** (m + 1))

if __name__ == "__main__":
    #!-------------------
    #! Section 4.1
    #!-------------------
    tic = time.time()
    figure()
    demo_distr(UniformDistr(0,1) * UniformDistr(0,1), theoretical = lambda x: -log(x))
    figure()
    demo_distr(UniformDistr(0,1) / UniformDistr(0,1), theoretical = lambda x: (x<=1) * 0.5 + (x>1) * 0.5 / x**2)
    
    #! Section 4.4.1
    def prod_uni_pdf(n, x):
        pdf = (-log(x)) ** (n-1)
        for i in range(2, n):
            pdf /= i
        return pdf
    figure()
    demo_distr(UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1), theoretical = partial(prod_uni_pdf, 3))
    
    pu = UniformDistr(0,1)
    for i in range(4):
        pu *= UniformDistr(0,1)
    figure()
    demo_distr(pu, theoretical = partial(prod_uni_pdf, 5))
    for i in range(6):
        pu *= UniformDistr(0,1)
    figure()
    demo_distr(pu, theoretical = partial(prod_uni_pdf, 11))
    for i in range(10):
        pu *= UniformDistr(0,1)
    figure()
    demo_distr(pu, theoretical = partial(prod_uni_pdf, 21))
    figure()
    pu += pu
    demo_distr(pu)
    
    #! Example 4.4.1
    figure()
    demo_distr(BetaDistr(9, 3) * BetaDistr(8, 3) * BetaDistr(4, 2),
               theoretical = lambda x: (3960.0/7*x**3 - 1980*x**4 + 99000*x**7 +
                                        (374220+356400*log(x))*x**8 - (443520 - 237600*log(x))*x**9 - 198000.0/7*x**10))
    #! Example 4.5.1
    figure()
    demo_distr(NormalDistr(0,1) * NormalDistr(0,1))
    np = NormalDistr(0,1)
    for i in range(5):
        np *= NormalDistr(0,1)
    figure()
    demo_distr(np, xmin = -0.3, xmax = 0.3)
    
    for i in range(10):
        np *= NormalDistr(0,1)
    figure()
    demo_distr(np, xmin = -0.3, xmax = 0.3, ymax = 5)
    
    #! Figure 4.6.1
    #! 
    #! the actual figure is slightly different, Springer doesn't give
    #! exact parameters
    figure()
    demo_distr(NormalDistr(1,1) / NormalDistr(1,0.5), xmin = -4, xmax = 4)
    
    #! Exercise 4.1
    def prod_uni_pdf_a(a, n, x):
        pdf = 0.5 * (n*log(a)-log(abs(x))) ** (n-1)
        for i in range(2, n):
            pdf /= i
        pdf *= a**(-n)
        return pdf
    figure()
    demo_distr(UniformDistr(-0.2,0.2) * UniformDistr(-0.2,0.2), theoretical = partial(prod_uni_pdf_a, 0.2, 2))
    
    figure()
    demo_distr(UniformDistr(-2,2) * UniformDistr(-2,2) * UniformDistr(-2,2), theoretical = partial(prod_uni_pdf_a, 2, 3))
    
    pu = UniformDistr(-1.5,1.5)
    for i in range(4):
        pu *= UniformDistr(-1.5,1.5)
    figure()
    demo_distr(pu, theoretical = partial(prod_uni_pdf_a, 1.5, 5))
    
    #! Exercise 4.2
    figure()
    from numpy import zeros_like, asfarray
    def prod_uni_uni(mu1, mu2, x):
        assert mu1 > 0.5 and mu2 > 0.5 and mu1 > mu2
        y = zeros_like(asfarray(x))
        mask = (x > (mu1-0.5)*(mu2+0.5)) & (x <= (mu1+0.5)*(mu2+0.5))
        y[mask] = -log(x[mask] / (mu1+0.5) / (mu2+0.5))
        mask = (x > (mu1+0.5)*(mu2-0.5)) & (x <= (mu1-0.5)*(mu2+0.5))
        y[mask] = -log((mu1-0.5)/(mu1+0.5))
        mask = (x > (mu1-0.5)*(mu2-0.5)) & (x <= (mu1+0.5)*(mu2-0.5))
        y[mask] = -log((mu1-0.5) * (mu2-0.5) / x[mask])
        return y
    demo_distr(UniformDistr(1, 2) * UniformDistr(0.5, 1.5), theoretical = partial(prod_uni_uni, 1.5, 1))
    figure()
    demo_distr(UniformDistr(1.5, 2.5) * UniformDistr(0.5, 1.5), theoretical = partial(prod_uni_uni, 2, 1))
    figure()
    demo_distr(UniformDistr(25, 26) * UniformDistr(14, 15), theoretical = partial(prod_uni_uni, 25.5, 14.5))
    
    #! Exercise 4.3
    figure()
    demo_distr(CauchyDistr() * UniformDistr(0,1), xmin = -3, xmax = 3)
    figure()
    demo_distr(CauchyDistr() * UniformDistr(0,1) + CauchyDistr() * UniformDistr(0,1), xmin = -3, xmax = 3)
    from numpy import pi, log1p
    def prodcauchy_uni_pdf(a, gamma, x):
        return 0.5/(pi*a*gamma)*log1p((a*a*gamma*gamma/(x*x)))
    figure()
    demo_distr(CauchyDistr() * UniformDistr(-1,1), theoretical = partial(prodcauchy_uni_pdf, 1, 1), xmin = -3, xmax = 3, ymax = 5)
    figure()
    demo_distr(CauchyDistr(gamma = 0.1) * UniformDistr(-3,3), theoretical = partial(prodcauchy_uni_pdf, 3, 0.1), xmin = -3, xmax = 3, ymax = 5)
    figure()
    demo_distr(CauchyDistr(gamma = 10) * UniformDistr(-0.3,0.3), theoretical = partial(prodcauchy_uni_pdf, 0.3, 10), xmin = -3, xmax = 3, ymax = 5)
    
    #! Exercise 4.4
    figure()
    # TODO: add theoretical (needs incomplete gamma function)
    demo_distr(UniformDistr(-2.1,2.1) * GammaDistr(2,2), xmin = -40, xmax = 40)
    
    #! Exercise 4.5
    from numpy import pi
    def prod_cauchy(gamma, x):
        return gamma**2 / (pi*pi*(x**2 - gamma**4)) * log(x**2 / gamma**4)
    figure()
    demo_distr(CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1), xmin = -20, xmax = 20, ymax = 1.5, theoretical = partial(prod_cauchy, 1.0))
    figure()
    demo_distr(CauchyDistr(gamma = 3) * CauchyDistr(gamma = 3), xmin = -20, xmax = 20, ymax = 1.5, theoretical = partial(prod_cauchy, 3.0))
    
    #! Exercise 4.5
    figure()
    demo_distr(CauchyDistr(gamma = 1) / CauchyDistr(gamma = 1), xmin = -20, xmax = 20, ymax = 1.5, theoretical = partial(prod_cauchy, 1.0))
    figure()
    c = NormalDistr() / NormalDistr()
    c2 = NormalDistr() / NormalDistr()
    demo_distr(c / c2, xmin = -20, xmax = 20, ymax = 1.5, theoretical = partial(prod_cauchy, 1.0))
    figure()
    demo_distr(CauchyDistr(gamma = 3) / CauchyDistr(gamma = 3), xmin = -20, xmax = 20, ymax = 1.5, theoretical = partial(prod_cauchy, 1.0))
       
    #! Exercise 4.7
    from numpy import pi
    def prod_3cauchy(gamma, x):
        return gamma**3 / (2*pi**3*(x**2 + gamma**6)) * (log(x**2 / gamma**6)**2 + pi**2)
    figure()
    demo_distr(CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1), xmin = -20, xmax = 20, ymax = 1.5,
               theoretical = partial(prod_3cauchy, 1.0))
    figure()
    cp = CauchyDistr(gamma = .1) * CauchyDistr(gamma = .1) * CauchyDistr(gamma = .1)
    demo_distr(cp, xmin = -20, xmax = 20, ymax = 1.5,
               theoretical = partial(prod_3cauchy, .1))
       
    #! Exercise 4.8
    from numpy import pi
    def prod_5cauchy(gamma, x):
        return gamma**5 / (24*pi**5*(x**2 + gamma**10)) * (log(x**2 / gamma**10)**4 + 10*pi**2*log(x**2 / gamma**10)**2 + 9*pi**4)
    figure()
    demo_distr(CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1) * CauchyDistr(gamma = 1),
               xmin = -20, xmax = 20, ymax = 1.5,
               theoretical = partial(prod_5cauchy, 1.0))
    
    #! Exercise 4.9
    from numpy import pi
    def prod_10cauchy(gamma, x):
        return gamma**10 / (362880*pi**10*(x**2 - gamma**20)) * (log(x**2 / gamma**20)**9
                                                                 + 120*pi**2*log(x**2 / gamma**20)**7
                                                                 + 4368*pi**4*log(x**2 / gamma**20)**5
                                                                 + 52480*pi**6*log(x**2 / gamma**20)**3
                                                                 + 147456*pi**8*log(x**2 / gamma**20))
    c = CauchyDistr()
    for i in range(9):
        c *= CauchyDistr()
    figure()
    demo_distr(c, xmin = -20, xmax = 20, ymax = 1.5,
               theoretical = partial(prod_10cauchy, 1.0))
    
    #! Exercise 4.11
    figure()
    demo_distr(BetaDistr(5, 2) * BetaDistr(6, 2) * BetaDistr(6, 3), xmin = 0, xmax = 1,
               theoretical = lambda x: 17640*(2*x**4-33*x**5-18*x**5*log(x)-6*x**5*log(x)**2-12*x**6*log(x)+30*x**6+x**7))
    
    #! Exercise 4.12
    def log_prod_uni(n):
        u = UniformDistr(0,1)
        for i in range(n-1):
            u *= UniformDistr(0,1)
        return -2 * log(u)
    figure()
    demo_distr(log_prod_uni(2), theoretical = ChiSquareDistr(4))
    figure()
    demo_distr(log_prod_uni(3), theoretical = ChiSquareDistr(6))
    figure()
    demo_distr(log_prod_uni(5), theoretical = ChiSquareDistr(10))
    
    #! Exercise 4.13
    # TODO: compare with K_0
    figure()
    demo_distr(NormalDistr() * NormalDistr(), xmin = -3, xmax = 3, ymax = 2)
    
    #! Exercise 4.14
    def f_distr_test(df1, df2, mode = 1):
        #num = ZeroDistr()
        #for i in xrange(df1):
        #    num += NormalDistr(0,1) ** 2
        #den = ZeroDistr()
        #for i in xrange(df2):
        #    den += NormalDistr(0,1) ** 2
        num = NormalDistr(0, 1) ** 2
        for i in range(df1 - 1):
            num += NormalDistr(0,1) ** 2
        den = NormalDistr() ** 2
        for i in range(df2 - 1):
            den += NormalDistr(0,1) ** 2
        if mode == 1:
            num /= df1
            den /= df2
        return num, den, num / den
    
    from pacal.utils import lgamma
    def f_disrt_v2(df1, df2, x):
        df1 = float(df1)
        df2 = float(df2)
        norm = exp(lgamma((df1 + df2) / 2) - lgamma(df1 / 2) - lgamma(df2 / 2))
        y = norm * x ** (df1 / 2 - 1) / (x + 1) ** ((df1 + df2) / 2)
        return y
    for mode in [1, 2]:
        for df1, df2 in [(1,1), (2,2), (3,1), (4,5)]:
            num, den, f = f_distr_test(df1, df2, mode)
            if mode == 1:
                num *= df1
                den *= df2
            #figure()
            #demo_distr(num, theoretical = ChiSquareDistr(df1))
            #figure()
            #demo_distr(den, theoretical = ChiSquareDistr(df2))
            if df2 < 2:
                ymax = 3
            else:
                ymax = None
            if mode == 1:
                theor = FDistr(df1, df2)
            else:
                theor = partial(f_disrt_v2, df1, df2)
            figure()
            demo_distr(f, theoretical = theor, xmax = 10, ymax = ymax)
    
    #! Exercise 4.15
    def student_t(df):
        den = NormalDistr() ** 2
        for i in range(df - 1):
            den += NormalDistr() ** 2
        return NormalDistr() / sqrt(den / df)
    for df in [1, 2, 3, 5]:
        figure()
        if df < 2:
            ymax = 3
        else:
            ymax = None
        demo_distr(student_t(df), theoretical = StudentTDistr(df), xmax = 10, ymax = ymax)
    
    
    #! Exercise 4.16
    figure()
    demo_distr(ExponentialDistr(0.5) / ExponentialDistr(0.5), theoretical = lambda x: (1 + x)**-2)
    
    #! Exercise 4.17
    for l in [0.1, 1, 10]:
        d = (ExponentialDistr(l) + ExponentialDistr(l)) / (ExponentialDistr(l) + ExponentialDistr(l))
        figure()
        demo_distr(d, xmax = 10)
       
    #! Exercise 4.18
    def theor_beta_prod(alpha, n, x):
        norm = float((alpha + 1) ** n)
        for i in range(1, n):
            norm /= i
        return norm * x ** alpha * log(1.0 / x) ** (n - 1)
    for alpha in [0.2, 1, 2, 4]:
        for n in [2, 3, 5]:
            d = BetaDistr(alpha + 1, 1)
            for i in range(n - 1):
                d *= BetaDistr(alpha + 1, 1)
            figure()
            demo_distr(d, theoretical = partial(theor_beta_prod, alpha, n))
    
    #! Exercise 4.19
    def theor_quot_power(alpha, x):
        return (x > 0) * (x <= 1) * (alpha + 1.0) / 2 * x**alpha + (x > 1) * (alpha + 1.0) / 2 * x**(-alpha-2)
    for alpha in [0.1, 1, 10]:
        figure()
        demo_distr(BetaDistr(alpha + 1, 1) / BetaDistr(alpha + 1, 1), theoretical = partial(theor_quot_power, alpha))
    
    #! Exercise 4.20
    def theor_quot_power2(a1, b1, a2, b2, m_plus_1, x):
        m = m_plus_1 - 1
        Ax = m_plus_1**2*x**m / ((b1*b2)**m_plus_1 - (b1*a2)**m_plus_1 - (b2*a1)**m_plus_1 + (a1*a2)**m_plus_1)
        y = ((b1*a2 <= x) * (x <= b1*b2) * (-Ax) * log(x/b1/b2) +
             (a1*b2 <= x) * (x < b1*a2) * Ax * (-log(x/b1/b2)+log(x/b1/a2)) +
             (a1*a2 <= x) * (x < a1*b2) * Ax * (-log(x/b1/b2)+log(x/b1/a2) + log(x/a1/b2))
             )
        return y
    for a1, b1, a2, b2, m in [(1.1, 4, 1, 3, 1),
                              (1.1, 45, 0.9, 5, 1)]:
        assert a1 >= 0 and a2 >= 0
        assert b1 > b2 > a1 > a2
        assert b1*b2 > b1*a2 > b2*a1 > a1*a2
        partial
#        d1 =  FunDistr(lambda x: (m+1)*x / (b1**(m+1) - a1**(m+1)), breakPoints=[a1, b1])
#        d2 =  FunDistr(lambda x: (m+1)*x / (b2**(m+1) - a2**(m+1)), breakPoints=[a2, b2])
        d1 =  FunDistr(partial(funEx4_20,m,a1,b1), breakPoints=[a1, b1])
        d2 =  FunDistr(partial(funEx4_20,m,a2,b2), breakPoints=[a2, b2])
        figure()
        demo_distr(d1 * d2, theoretical = partial(theor_quot_power2, a1, b1, a2, b2, m+1))
    
    #! Exercise 4.24
    try:
        from scipy.special import exp1
        have_exp1 = True
        def theor_prod_uni_norm(a, sigma, x):
            return 0.5/sqrt(2*numpy.pi) * exp1(0.5 * (x/(a*sigma))**2)
    except ImportError:
        have_exp1 = False
        
    for a, sigma in [(1,1), (10,0.1), (0.1,10)]:
        d = UniformDistr(-a, a) * NormalDistr(0, sigma)
        figure()
        if have_exp1:
            demo_distr(d, theoretical = partial(theor_prod_uni_norm, a, sigma))
        else:
            demo_distr(d)
    
    #! Exercise 4.25
    from pacal.utils import lgamma
    def theor_quot_gamma(b1, b2, x):
        lnorm = lgamma(b1 + b2) - lgamma(b1) - lgamma(b2)
        return x**(b1-1)*(1+x)**(-b1-b2)*exp(lnorm)
    for b1, b2 in [(1,1), (2, 7), (3, 3)]:
        d = GammaDistr(b1, 1) / GammaDistr(b2, 1)
        figure()
        demo_distr(d, xmin = 0, xmax = 10, theoretical = partial(theor_quot_gamma, b1, b2))
    
    #! Exercise 4.26
    def theor_quot_triang(x):
        y = ((x <= 0.5) * x * 7.0/6
             + (x > 0.5)*(x<=1)*(8.0/3 -1.5*x -2.0/(3*x**2) +1.0/(6*x**3))
             + (x > 1)*(x<=2)  *(-2.0/3 +x/6.0 +8.0/(3*x**2) -3.0/(2*x**3))
             + (x > 2) * 7.0/(6*x**3))
        return y
    T1 = UniformDistr(0,1) + UniformDistr(0,1)
    T2 = UniformDistr(0,1) + UniformDistr(0,1)
    V = T1 / T2
    demo_distr(V, theoretical = theor_quot_triang)
    
    #! Exercise 4.27
    #! Works, implemented elsewhere
    
    #! Exercise 4.28
    def norm_ratio_pdf(mu1, sigma1, mu2, sigma2, rho, x):
        a = sqrt((x/sigma1)**2 - 2*rho*x/(sigma1*sigma2) + 1.0/sigma2**2)
        b = mu1*x/sigma1**2 - rho*(mu1 + mu2*x)/(sigma1*sigma2) + mu2/sigma2**2
        c = mu1**2/sigma1**2 - 2*rho*mu1*mu2/(sigma1*sigma2) + mu2**2/sigma2**2
        d = exp((b**2 - c*a**2) / (2*(1-rho**2)*a**2))
        try:
            from scipy.stats.distributions import norm
            def norm_cdf(z):
                return norm.cdf(z)
        except:
            N = NormalDistr()
            def norm_cdf(z):
                return N.cdf(z)
        #! WARNING: a bug in Springer's book in the next line
        y =  b*d/(sqrt(2*numpy.pi)*sigma1*sigma2*a**3) * (norm_cdf(b/(sqrt(1-rho**2)*a)) - norm_cdf(-b/(sqrt(1-rho**2)*a)))
        y += sqrt(1-rho**2)/(numpy.pi*sigma1*sigma2*a**2) * exp(-c/(2*(1-rho**2)))
        return y
    for mu1, sigma1, mu2, sigma2, rho in [(0.0,1.0,0.0,1.0,0.0),
                                          (0.1,1.0,0.1,1.0,0.0),
                                          (1.0,1.0,1.0,1.0,0),
                                          (5.0,1.0,5.0,1.0,0.0),
                                          (50.0,1.0,50.0,1.0,0.0),
                                          (2.0,1.0,-50.0,2.0,0.0),
                                          ]:
        d = NormalDistr(mu1, sigma1) / NormalDistr(mu2, sigma2)
        figure()
        demo_distr(d, theoretical = partial(norm_ratio_pdf, mu1, sigma1, mu2, sigma2, rho))
    
    #! Exercise 4.29
    for A, B, q, p in [(1, 1, 3, 7),
                       (1, 1, 0.5, 5),
                       ]:
        assert A > 0
        assert B > 0
        assert q < p
        u = GammaDistr(p, 1)
        v = BetaDistr(q, p-q)
        #! part a)
        figure()
        d = u*v
        demo_distr(d, theoretical = GammaDistr(q, 1))
        #! part b)
        if q == 0.5:
            figure()
            demo_distr(sqrt(2*d), theoretical = abs(NormalDistr(0, 1)))
    #! part c)
    figure()
    d = GammaDistr(2, 1) * UniformDistr(0, 1)
    demo_distr(d, theoretical = ExponentialDistr(1))
    
    #! Exercise 4.30
    #! TODO
    
    #! Exercise 4.31
    from pacal.utils import binomial_coeff
    for n, m in [(2,2), (5,7)]:
        N = UniformDistr(0,1)
        for i in range(n-1):
            N += UniformDistr(0,1)
        D = UniformDistr(0,1)
        for i in range(m-1):
            D += UniformDistr(0,1)
        d = N/D
        figure()
        demo_distr(d, xmax = 10)
    
        a = 0.9
        nmfact = 1
        for i in range(2, n+m+1):
            nmfact *= i
        nrm = a**m * nmfact
        s = 0
        for i in range(int(numpy.floor(m*a)+1)):
            for j in range(int(numpy.floor((m*a-i)/a)+1)):
                s += (-1)**(i+j) * binomial_coeff(n, i) * binomial_coeff(m, j) * ((m-j)*a-i)**(n+m)
        cdf_formula = s / nrm
        cdf_pacal = d.cdf(a)
        print("n={0},m={1}  cdf({2})={3}, err = {4}".format(n, m, a, cdf_pacal, abs(cdf_pacal - cdf_formula)))
    
    #! Exercise 4.32
    for l1, l2 in [(1,1), (10,1), (1,10)]:
        d = ExponentialDistr(l1) / ExponentialDistr(l2)
        figure()
        alpha = float(l2) / l1
        demo_distr(d, xmax = 10, theoretical = lambda x: alpha / (x+alpha)**2)
    print("time=", time.time() - tic)
    show()
