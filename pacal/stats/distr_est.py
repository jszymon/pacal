from __future__ import print_function

import inspect
from pacal import MixDistr
from numpy import prod, array, nan_to_num, ones, log
from scipy.optimize import fmin

class LoglikelihoodEstimator(object):
    def __init__(self, distr=None, xi=None, params=None, defvals=None, parconstr={}, debug_info=False):
        self.xi = xi
        self.parconstr = parconstr
        self.debug_info = debug_info
        if params is None:
            argspec = inspect.getargspec(distr.__dict__["__init__"])
            argspec.args.remove("self")
            self.params = argspec.args
            nargs = len(self.params)
            self.defvals = argspec.defaults[0:nargs]
        else:
            self.params = params
            self.defvals = defvals
        #print "params=", self.params
        #print "defvals=", self.defvals
        self.distr = distr
        self.parkvargs = {}
        i = 0
        for i in range(len(self.params)):
            self.parkvargs[self.params[i]] = self.defvals[i]
        if self.debug_info:  print("parkvargs=", self.parkvargs)
    def make_kwargs(self, params, vals):
        i = 0
        parkvargs = {}
        for i in range(len(params)):
            parkvargs[params[i]] = vals[i]
        return parkvargs
    def logli(self, parvals):
        kwargs = self.make_kwargs(self.params, parvals)
        #self.distr(**kwargs).summary()
        ll=-sum(log(self.distr(**kwargs).get_piecewise_pdf()(self.xi)+1e-300))
        if self.debug_info: print(parvals, ll, self.distr(**kwargs).get_piecewise_pdf()(self.xi))
        return ll
    def find_params(self):
        paropt=fmin(self.logli, self.defvals)
        kwargs = self.make_kwargs(self.params,  paropt)
        if self.debug_info:
            self.distr(**kwargs).plot()
            hist(self.xi, bins=40, normed=True)
        return kwargs
    def __str__(self):
        return "Distr"

if __name__ == "__main__":
    from pylab import figure, show, hist
    from pacal.standard_distr import *

    a = LoglikelihoodEstimator(BetaDistr, xi=BetaDistr(1.4,4.1).rand(1000))
    b = LoglikelihoodEstimator(distr=NormalDistr, xi=NormalDistr(2,1).rand(1000))
    c = LoglikelihoodEstimator(distr=CauchyDistr, xi=CauchyDistr(1,1).rand(1000))
    d = LoglikelihoodEstimator(distr=ParetoDistr, xi=ParetoDistr(1.4).rand(1000))

    print(a.find_params())
    print(b.find_params())
    print(c.find_params())
    print(d.find_params())
    show()
