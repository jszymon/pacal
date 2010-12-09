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
X.summary()
Y.summary()

fun = lambda p: p*X + (1-p)*Y

pOptVar = fminbound(lambda p: fun(p).get_piecewise_pdf().var(), 0, 1, xtol = 1e-16)
print "pOptVar = ", pOptVar 
fun(pOptVar).summary()

pOptMad= fminbound(lambda p: fun(p).get_piecewise_pdf().medianad(), 0, 1, xtol = 1e-16)
print "pOptVar = ", pOptMad 
fun(pOptMad).summary()

pOptIQrange = fminbound(lambda p: fun(p).get_piecewise_pdf().iqrange(0.05), 0, 1, xtol = 1e-16)
print "pOptIQrange = ", pOptIQrange 
fun(pOptIQrange).summary()

figure()
X.plot(color='k')
Y.plot(color='k')
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