"""Extreme value distribution demo."""

from pylab import figure, xlim, ylim, title, legend

from numpy import pi, tan

from pacal import *
from pacal.distr import demo_distr

EulConst = 0.5772156649015328606

colors = "kbgrcmy"
def extreme_limit_demo(X, ns, an, bn, limit_distr, xmin = None, xmax = None, ymax = None, **args):
    figure()
    title("Limit of maxima of " + X.getName())

    for i, n in enumerate(ns):
        Y = iid_max(X, n)
        Y = an(n) * (Y - bn(n))
        #demo_distr(Y, g)
        Y.plot(color=colors[i], label = "n=" + str(n))
        Y.summary()

    g.plot(linewidth=4, color = "m", alpha = 0.4, label = "limit")
    g.summary()

    if xmin is not None:
        xlim(xmin = xmin)
    if xmax is not None:
        xlim(xmax = xmax)
    ylim(ymin = 0)
    if ymax is not None:
        ylim(ymax = ymax)
    legend()


ns = [8, 16, 32, 64, 128]


X = UniformDistr(0,1)
an = lambda n: -n
bn = lambda n: 1
g = ExponentialDistr() # = WeibullDistr(1)
extreme_limit_demo(X, ns, an, bn, g, xmax = 10)

# TODO: find bn here
# X = ExponentialDistr()
# an = lambda n: 1
# bn = lambda n: log(n) - EulConst
# g = GumbelDistr()
# extreme_limit_demo(X, ns, an, bn, g)

X = NormalDistr()
an = lambda n: sqrt(2*log(n))
bn = lambda n: sqrt(2*log(n)) - 0.5 * (log(log(n)) + log(4*pi)) / sqrt(2*log(n))
g = GumbelDistr()
extreme_limit_demo(X, ns, an, bn, g)
 
X = CauchyDistr()
an = lambda n: 1.0 / tan(pi/2 - pi/n)
bn = lambda n: 0
g = FrechetDistr(1)
extreme_limit_demo(X, ns, an, bn, g)

show()
