"""Extreme value distribution demo."""

from __future__ import print_function

from pylab import figure, xlim, ylim, title, legend

from numpy import pi, tan

from pacal import *
from pacal.distr import demo_distr

if __name__ == "__main__":
    colors = "kbgrcmykbgrcmykbgrcmy"
    def extreme_limit_demo(X, ns, an, bn, limit_distr, xmin = None, xmax = None, ymax = None, **args):
        figure()
        title("Limit of maxima of " + X.getName())

        for i, n in enumerate(ns):
            Y = iid_max(X, n)
            Y = an(n) * (Y - bn(n))
            Y.plot(color=colors[i], label = "n=" + str(n), **args)
            Y.summary()

        limit_distr.plot(linewidth=4, color = "m", alpha = 0.4, label = "limit")
        limit_distr.summary()

        if xmin is not None:
            xlim(xmin = xmin)
        if xmax is not None:
            xlim(xmax = xmax)
        ylim(ymin = 0)
        if ymax is not None:
            ylim(ymax = ymax)
        legend()


    ns = [8, 16, 32, 64, 128, 256]#, 512, 1024]

    X = UniformDistr(0,1)
    an = lambda n: -n
    bn = lambda n: 1
    g = ExponentialDistr() # == WeibullDistr(1)
    extreme_limit_demo(X, ns, an, bn, g, xmax = 10, numberOfPoints = 50000)

    X = ExponentialDistr()
    an = lambda n: 1
    bn = lambda n: log(n)
    g = GumbelDistr()
    extreme_limit_demo(X, ns, an, bn, g, xmin = -2, xmax = 10)

    X = NormalDistr()
    invcdf = X.get_piecewise_invcdf()
    bn = lambda n: invcdf(1.0 - 1.0/n)
    an = lambda n: 1.0 / (invcdf(1.0 - 1.0/n/exp(1)) - invcdf(1.0 - 1.0/n))
    #an = lambda n: sqrt(2*log(n))
    #bn = lambda n: sqrt(2*log(n)) - 0.5 * (log(log(n)) + log(4*pi)) / sqrt(2*log(n))
    g = GumbelDistr(0, 1)
    extreme_limit_demo(X, ns, an, bn, g, xmin = -4, xmax = 8)

    X = CauchyDistr()
    an = lambda n: 1.0 / tan(pi/2 - pi/n)
    bn = lambda n: 0
    g = FrechetDistr(1)
    extreme_limit_demo(X, ns, an, bn, g, xmin = -1, xmax = 20, numberOfPoints = 50000)

    show()
