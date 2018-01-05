#!
#!------------------
#! CHAPTERS 6, 7, 8
#!------------------
#!
from __future__ import print_function

from functools import partial
import numpy

from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr


#! Exercise 6.3
d = NormalDistr() * NormalDistr() + NormalDistr() * NormalDistr()
figure()
demo_distr(d, theoretical = LaplaceDistr())

#! Example 7.3.1
w1 = WeibullDistr(2)
w2 = WeibullDistr(3)
figure()
demo_distr(w1 * w2)

#! Example 7.3.2
x1 = BetaDistr(9,3)
x2 = BetaDistr(8,3)
x3 = BetaDistr(4,2)
figure()
demo_distr(x1 * x2 * x3)

#! Example 8.6.1
x1 = abs(NormalDistr(0, 1.5))
x2 = GammaDistr(.2,1)
x3 = ExponentialDistr(1.0/0.4)
x4 = abs(NormalDistr(0, 2))
d = x1+x2*x3-5*x4
figure()
demo_distr(d, xmax=20)
exm = -6.7020187668243558
print("exact mean =", exm, "err =", d.mean() - exm)

#! Example 8.7.1
#!
#! poles at nonzero locations, not handled well yet
x1 = GammaDistr(2,0.4)
x2 = BetaDistr(2,0.5)
x3 = ExponentialDistr(1.0/0.4)
d = 0.25*x1*x2 + x3 + 7.21
figure()
demo_distr(d)

#! Example 8.13.1
d = BetaDistr(5,2) * BetaDistr(6,2) * BetaDistr(6,3)
figure()
demo_distr(d)

#! Example 8.13.2
d = abs(NormalDistr()) + ExponentialDistr()
figure()
demo_distr(d)

show()
