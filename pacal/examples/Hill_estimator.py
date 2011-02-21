#!===================================================
#! Distribution of Hill's tail exponent distribution
#!===================================================

from pacal import *
from pylab import figure, show

def Hill_estim_distr(d, n, xmin):
    """The distribution of Hill's estimator for given distribution d
    >= xmin and sample size n."""
    s = log(d / xmin)
    for i in xrange(n - 1):
        s += log(d / xmin)
    return 1 + n / s


a = Hill_estim_distr(ParetoDistr(2, 1), 5, 1)
a.summary()
a.plot()
show()

figure()
a = Hill_estim_distr(ParetoDistr(1, 1), 10, 1)
a.summary()
a.plot()
show()
