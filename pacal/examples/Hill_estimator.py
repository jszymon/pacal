#!===================================================
#! Distribution of Hill's tail exponent distribution
#!===================================================

from pacal import *
from pylab import figure, title, show

def Hill_estim_distr(d, n, xmin):
    """The distribution of Hill's estimator for given distribution d
    >= xmin and sample size n."""
    d = CondGtDistr(d, xmin)
    s = log(d / xmin)
    for i in xrange(n - 1):
        s += log(d / xmin)
    return 1 + n / s


a = Hill_estim_distr(ParetoDistr(2, 1), 5, 1)
a.summary()
a.plot()
title("Sample distribution of Hill's estimator on 5 Pareto(2) samples")
#show()

figure()
a = Hill_estim_distr(ParetoDistr(1, 1), 10, 2)
a.summary()
a.plot()
title("Sample distribution of Hill's estimator on 10 Pareto(1) samples")
show()
