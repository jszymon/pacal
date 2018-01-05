#!
#!-----------------------
#! CHAPTER 5 - FUNCTIONS
#!-----------------------
#!
from __future__ import print_function

from functools import partial
import numpy
import time

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr

if __name__ == "__main__":
        
    tic = time.time()
    #! Example 5.1.3
    d = NormalDistr() + NormalDistr() * NormalDistr()
    demo_distr(d)
    
    #! Example 5.5
    d = ExponentialDistr() / (ExponentialDistr() + ExponentialDistr())
    figure()
    demo_distr(d, xmax=20, ymax=1.5)
    
    #! Exercise 5.5
    #! part a
    figure()
    demo_distr(NormalDistr() / sqrt((NormalDistr()**2 + NormalDistr()**2) / 2), xmin=-3, xmax=3)
    #! part b
    figure()
    demo_distr(2 * NormalDistr()**2 / (NormalDistr()**2 + NormalDistr()**2), xmax=20, ymax=2)
    #! part c
    figure()
    demo_distr(3 * NormalDistr()**2 / (NormalDistr()**2 + NormalDistr()**2 + NormalDistr()**2), xmax=20, ymax=2)
    #! part d
    figure()
    demo_distr((NormalDistr()**2 + NormalDistr()**2) / (NormalDistr()**2 + NormalDistr()**2), xmax=20)
    
    #! Exercise 5.6
    d = sqrt(UniformDistr(0,1)**2 + UniformDistr(0,1)**2)
    # a bug in Springer??
    def theor_ampl_uni(x):
        return (x<1)*numpy.pi/2*x + (x>=1)*(2*numpy.arcsin(1.0/x) - 0*numpy.pi/2)
    figure()
    #demo_distr(d, theoretical = theor_ampl_uni, histogram=True)
    demo_distr(d)
    print("time=", time.time() - tic)
    show()
