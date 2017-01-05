#!
#!-------------------
#! CHAPTER 3 - SUMS
#!-------------------
#!
from __future__ import print_function

from functools import partial
import numpy

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr
import time
if __name__ == "__main__":
        
    tic  =time.time()
    #! Figure 3.1.1
    figure()
    demo_distr(UniformDistr(0,1) + UniformDistr(0,1),          
                theoretical = lambda x: x * ((x >= 0) & (x < 1)) + (2-x) * ((x >= 1) & (x <= 2)))
    #! Figure 3.1.2
    figure()
    demo_distr(UniformDistr(0,1) - UniformDistr(0,1),
               theoretical = lambda x: (x+1) * ((x >= -1) & (x < 0)) + (1-x) * ((x >= 0) & (x <= 1)))
    
    #!-------------------
    #! Section 3.2.2
    #!-------------------
    figure()
    demo_distr(ChiSquareDistr(1) + ChiSquareDistr(1),
              theoretical = ExponentialDistr(0.5))
    figure()
    demo_distr(ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1),
              theoretical = ChiSquareDistr(3))
    figure()
    demo_distr(ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1),
              theoretical = ChiSquareDistr(4))
    figure()
    demo_distr(ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1) + ChiSquareDistr(1),
              theoretical = ChiSquareDistr(5))
    figure()
    demo_distr(ChiSquareDistr(1) + ChiSquareDistr(3),
               theoretical = ChiSquareDistr(4))
    figure()
    demo_distr(ChiSquareDistr(1) + (ChiSquareDistr(2)+ChiSquareDistr(1)),
               theoretical = ChiSquareDistr(4))
    figure()
    demo_distr(ChiSquareDistr(2) + ChiSquareDistr(2),
               theoretical = ChiSquareDistr(4))
    figure()
    demo_distr((ChiSquareDistr(1)+ChiSquareDistr(1)) + (ChiSquareDistr(1)+ChiSquareDistr(1)),
               theoretical = ChiSquareDistr(4))
    figure()
    demo_distr(ChiSquareDistr(10) + ChiSquareDistr(11),
               theoretical = ChiSquareDistr(21))
    cd = ChiSquareDistr(4)
    for i in range(17):
        cd = cd + ChiSquareDistr(1)
        #print i, cd.pdf(1)
    figure()
    demo_distr(cd, theoretical = ChiSquareDistr(21))
    
    figure()
    demo_distr(ChiSquareDistr(1000) + ChiSquareDistr(101),
               theoretical = ChiSquareDistr(1101))
    
    #!-------------------
    #! Section 3.3.1
    #!-------------------
    figure()
    demo_distr(UniformDistr(0,1) + ExponentialDistr(),
               theoretical = lambda x: -numpy.exp(-x) + (x >= 0)*(x <= 1) + (x > 1)*numpy.exp(-x+1))
    #! Exercise 3.10
    figure()
    demo_distr(ExponentialDistr() - ExponentialDistr(),
               theoretical = lambda x: numpy.exp(-numpy.abs(x)) / 2)
    
    #! Exercise 3.11
    figure()
    demo_distr(CauchyDistr() + CauchyDistr(), theoretical = CauchyDistr(gamma = 2))
    figure()
    demo_distr(CauchyDistr(center = -100) + CauchyDistr(center = 95), theoretical = CauchyDistr(gamma = 2, center = -5))
    figure()
    demo_distr(CauchyDistr(gamma = 10) + CauchyDistr(gamma = 50), theoretical = CauchyDistr(gamma = 60))
    figure()
    c = CauchyDistr(center = 1)
    for i in range(9):
        c += CauchyDistr()
    demo_distr(c, theoretical = CauchyDistr(gamma = 10, center = 1))
       
    #! Exercise 3.12
    #! Exact formula for sum of 'N' uniform random variables
    #!
    #! 'warning': this formula is very inaccurate for large n!
    #! much worse than our results!
    from numpy import ceil, isscalar, zeros_like, asfarray
    def uniform_sum_pdf(n, xx):
        if isscalar(xx):
            xx = asfarray(xx)
        y = zeros_like(asfarray(xx))
        for j, x in enumerate(xx):
            r = int(ceil(x))
            if r <= 0 or r > n:
                y[j] = 0
            else:
                nck = 1
                pdf = 0.0
                for k in range(r):
                    pdf += (-1)**k * nck * (x-k)**(n-1)
                    nck *= n - k
                    nck /= k + 1
                for i in range(2, n):
                    pdf /= i
                y[j] = pdf
        return y
    #!
    #!
    u = UniformDistr(0,1) + UniformDistr(0,1)
    figure()
    demo_distr(u, theoretical = partial(uniform_sum_pdf, 2))
    for i in range(2):
        u += UniformDistr(0,1)
    figure()
    demo_distr(u, theoretical = partial(uniform_sum_pdf, 3+i))
    
    u = UniformDistr(0,1)
    for i in range(49):
        u += UniformDistr(0,1)
    figure()
    demo_distr(u, theoretical = partial(uniform_sum_pdf, i+2))
    
    
    #! Exercise 3.15
    figure()
    demo_distr(ExponentialDistr() + BetaDistr(1,1))
    #! Exercise 3.16
    figure()
    demo_distr(UniformDistr(0,1) + ExponentialDistr() + ChiSquareDistr(4),
               theoretical = lambda x: (x>=0) * ((x<=1) + (x>1)*(exp(-x+1) + (x-1)*exp(-(x-1)/2)) - exp(-x) - x*exp(-x/2)))
    #! Exercise 3.18
    figure()
    demo_distr(UniformDistr(-1,0) + ExponentialDistr(),
               theoretical = lambda x: -numpy.exp(-(x+1)) + (x >= -1)*(x <= 0) + (x > 0)*numpy.exp(-x))
    #! Exercise 3.19
    figure()
    demo_distr(UniformDistr(-1,0) + ExponentialDistr() + NormalDistr(0, 1.0/numpy.sqrt(2)))
    print("time=", time.time() - tic)
    show()
