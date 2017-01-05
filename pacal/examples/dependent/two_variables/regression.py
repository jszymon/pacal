"""Regression of one variable onto another in a joint two-variable
distribution."""

from __future__ import print_function

from functools import partial

from numpy import linspace, isscalar, zeros_like, NaN, concatenate
from pylab import plot, figure, legend, title

from pacal import *
from pacal.utils import epsunique
from pacal.segments import PiecewiseFunction

def plot_regression(F, Ybreaks = None):
    assert F.d == 2
    X = F.marginals[0]
    Y = F.marginals[1]
    if Ybreaks is None:
        Ybreaks = Y.get_piecewise_pdf().getBreaks()

    def cond_pdf(F, Xpdf, x, y):
        if not isscalar(y):
            x = zeros_like(y) + x
        return F.pdf(x, y) / Xpdf
    def regfun(type, x):
        # value of regression functions at point x
        if isscalar(x):
            Xpdf = float(X.pdf(x))
            if Xpdf == 0:
                y = NaN
            else:
                distr = FunDistr(fun = partial(cond_pdf, F, Xpdf, x),
                                 breakPoints = Ybreaks)
                if type==1: y = distr.mean()
                if type==2: y = distr.median()
                if type==3: y = distr.mode()  
                if y is None:
                    y = NaN
        else:
            y = zeros_like(x)
            for i in range(len(x)):
                y[i] = regfun(type, x[i])
        return y 

    F.contour()

    Xbreaks = X.get_piecewise_pdf().getBreaks()
    Xbreaks = concatenate([Xbreaks, [F.a[0], F.b[0]]])
    Xbreaks.sort()
    Xbreaks = epsunique(Xbreaks)
    mreg = PiecewiseFunction(fun=partial(regfun, 1), breakPoints=Xbreaks).toInterpolated()
    mreg.plot(label = "mean")
    mreg = PiecewiseFunction(fun=partial(regfun, 2), breakPoints=Xbreaks).toInterpolated()
    mreg.plot(label = "median", color = "g")
    mreg = PiecewiseFunction(fun=partial(regfun, 3), breakPoints=Xbreaks).toInterpolated()
    mreg.plot(label = "mode", color = "r")
    legend()


print("bivariate normal...")
F = NDNormalDistr([0, 0], [[1, 0.5], [0.5, 1]])
plot_regression(F, Ybreaks = [-Inf, -5, -1, 1, 5, Inf])
title("bivariate normal, rho = 0.5")

figure()
print("Clayton copula...")
X, Y = BetaDistr(2,3), UniformDistr() + UniformDistr()
X.setSym("X"); Y.setSym("Y")
F = ClaytonCopula(theta = 0.5, marginals=[X, Y])
plot_regression(F)
title("Clayton copula, theta = 0.5")

figure()
print("Frank copula...")
F = FrankCopula(theta = 8, marginals=[X, Y])
plot_regression(F)
title("Frank copula, theta = 8")

show()
