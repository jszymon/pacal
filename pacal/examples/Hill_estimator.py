#!===================================================
#! Distribution of Hill's tail exponent distribution
#!===================================================

from __future__ import print_function

from pacal import *
from pylab import figure, title, show

if __name__ == "__main__":
    def Hill_estim_distr(d, n, xmin):
        """The distribution of Hill's estimator for given distribution d
        >= xmin and sample size n."""
        #d = CondGtDistr(d, xmin)
        d = d | Gt(xmin)
        s = iid_sum(log(d / xmin), n)
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
