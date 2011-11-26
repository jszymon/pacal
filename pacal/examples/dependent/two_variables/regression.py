"""Regression of one variable onto another in a joint two-variable
distribution."""

from functools import partial

from numpy import linspace, isscalar, zeros_like
from pylab import plot, figure, legend

from pacal import *
from pacal.segments import PiecewiseFunction

def plot_regression(F):
    def _fun(type, x):
        # value of regression functions at point
        if isscalar(x):
            distr = FunDistr(fun=lambda y: F.pdf(x,y) / X.pdf(x), breakPoints=Y.get_piecewise_pdf().getBreaks())
            if type==1: return distr.mean()
            if type==2: return distr.median()
            if type==3: return distr.mode()        
        else:
            y = zeros_like(x)
            for i in range(len(x)):
                y[i] = _fun(type, x[i])
            return y 

    #F.plot()
    figure()
    F.contour()

    mreg = PiecewiseFunction(fun=partial(_fun, 1), breakPoints=X.get_piecewise_pdf().getBreaks()).toInterpolated()
    mreg.plot(label = "mean")
    mreg = PiecewiseFunction(fun=partial(_fun, 2), breakPoints=X.get_piecewise_pdf().getBreaks()).toInterpolated()
    mreg.plot(label = "median", color = "g")
    mreg = PiecewiseFunction(fun=partial(_fun, 3), breakPoints=X.get_piecewise_pdf().getBreaks()).toInterpolated()
    mreg.plot(label = "mode", color = "r")
    legend()

    #figure()
    #distr = FunDistr(fun=lambda y: F.pdf(0.04,y)/X.pdf(0.04), breakPoints=Y.get_piecewise_pdf().getBreaks())
    #distr.summary()
#distr.plot()

#X, Y = BetaDistr(3, 6, sym="X"), BetaDistr(3, 1, sym="Y")
X, Y = BetaDistr(2,3), UniformDistr() + UniformDistr()
X.setSym("X"); Y.setSym("Y")
F = ClaytonCopula(theta = 0.5, marginals=[X, Y])
#F = FrankCopula(theta = 8, marginals=[X, Y])

#F = NDNormalDistr([0,0], [[1,0.5],[0.5,1]])

plot_regression(F)

show()
