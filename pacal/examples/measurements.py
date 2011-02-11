#! Examples from Springer's Book #
#!==============================================
from functools import partial
import numpy

from scipy.optimize.optimize import fminbound

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr

X = UniformDistr(0,2)
#Y = BetaDistr(2,2) 
Y = BetaDistr(0.5,0.5) 
X = UniformDistr(-1,1)
Y = NormalDistr()
X.summary()
Y.summary()

def fun(p):
    p = numpy.squeeze(p)
    print p
    return p*X + (1-p)*Y

pOptVar = fminbound(lambda p: fun(p).get_piecewise_pdf().var(), 0, 1, xtol = 1e-16)
print "pOptVar = ", pOptVar 
dopt = fun(pOptVar)
dopt.summary()
print "iqrange(0.025)=", dopt.get_piecewise_pdf().iqrange(0.025)

pOptMad = fminbound(lambda p: fun(p).get_piecewise_pdf().medianad(), 0, 1, xtol = 1e-16)
print "pOptMAD = ", pOptMad 
dopt = fun(pOptMad)
dopt.summary()
print "iqrange(0.025)=", dopt.get_piecewise_pdf().iqrange(0.025)

pOptIQrange = fminbound(lambda p: fun(p).get_piecewise_pdf().iqrange(0.025), 0, 1, xtol = 1e-16)
print "pOptIQrange = ", pOptIQrange 
dopt = fun(pOptIQrange)
dopt.summary()
print "iqrange(0.025)=", dopt.get_piecewise_pdf().iqrange(0.025)

print "-----------------------"
figure()
#X.plot(color='k')
#Y.plot(color='k')
fun(pOptVar).plot(color='r')
fun(pOptMad).plot(color='g')
fun(pOptIQrange).plot(color='b')
figure()
X.get_piecewise_cdf().plot(color='k')
Y.get_piecewise_cdf().plot(color='k')
fun(pOptVar).get_piecewise_cdf().plot(color='r')
fun(pOptMad).get_piecewise_cdf().plot(color='g')
fun(pOptIQrange).get_piecewise_cdf().plot(color='b')
show()
