"""Demo of central limit theorem"""

from pylab import *

import sys
sys.path.append("../")
from distr import NormalDistr, UniformDistr, DivDistr, SumDistr, ConstMulDistr, SquareDistr
from distr import InterpolatedDistr
import distr

def plotdistr(d, l = 0, u = 1):
    X = linspace(l,u,300)
    Y = [d.pdf(x) for x in X]
    plot(X,Y)

from distr import set_plot__
def central_limit_demo(X, n = 5, l=0, u=1):
    Y = X
    for i in xrange(n):
        if i > 0:
            A = ConstMulDistr(Y, 1.0/(i+1))
        else:
            A = Y
        plotdistr(A, l, u)
        #set_plot__()
        if i < n - 1:
            Y = InterpolatedDistr(SumDistr(Y, X))
            print "error =", Y.err, "interp. nodes used =", Y.n_nodes, "#breakpoints =", len(Y.breaks)

if __name__ == "__main__":
    # uniform distribution
    X = UniformDistr(0,1); l,u=0,1
    # Chi^2_1
    #X = interpolatedDistr(squareDistr(NormalDistr(0,1))); l,u=-5,5
    #X = squareDistr(NormalDistr(0,1)); l,u=0,5
    #X = squareDistr(NormalDistr(0,1)); l,u=0,5
    # my favorite distribution
    X = InterpolatedDistr(DivDistr(UniformDistr(1,3), UniformDistr(-2,1))); l,u=-6,6
    # Cauchy
    #X = InterpolatedDistr(DivDistr(NormalDistr(0,1), NormalDistr(0,1))); l,u=-5,5
    
    central_limit_demo(X, 4, l, u)
    show()
