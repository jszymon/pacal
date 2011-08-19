#! Inaccurate places in PaCal 
#! ==========================

from pylab import *
from pacal import *
from pacal.distr import demo_distr

#! Here we describe inaccurate places in the PaCal project and future topics.

#! Exponential function and logarithm
#! ----------------------------------

#$  The real power or random variable $(X^\alpha)$ is always
#$  well defined, but logarithm and exponent may cause troubles. Bellow we have 
#$ some of examples:
#! 
figure()
Y = abs(NormalDistr()) ** NormalDistr()
demo_distr(Y, histogram = True, xmin = 0, xmax = 3, ymax = 3, hist_bins = 500)
Y.get_piecewise_cdf().plot(color='b', linewidth = 2.0) 
#! 
#! As one can see we obtain only six digits accuracy. The main problem
#! concerns singularity at point 1 such kind of singularity is difficult to 
#! detection and difficult to interpolation.
#! 
Y = UniformDistr(0,1) ** NormalDistr(1,1)
figure()
demo_distr(Y, histogram = True, xmin = 0, xmax = 3, ymax = 3, hist_bins = 500)
Y.get_piecewise_cdf().plot(color='b', linewidth = 2.0) 
#! 
Y = UniformDistr(0.5, 1.5) ** NormalDistr(0,1)
figure()
demo_distr(Y, histogram = True, xmin = 0, xmax = 3, ymax = 3, hist_bins = 500)
Y.get_piecewise_cdf().plot(color='b', linewidth = 2.0) 
#! 
Y = UniformDistr(0.0, 0.5) ** NormalDistr(3,1)
figure()
demo_distr(Y, histogram = True, xmin = 0, xmax = 3, ymax = 3, hist_bins = 500)
Y.get_piecewise_cdf().plot(color='b', linewidth = 2.0) 
#! 
Y = UniformDistr(0.5, 2.0) ** UniformDistr(0.5,2)
figure()
demo_distr(Y, histogram = True, xmin = 0, xmax = 4, hist_bins = 500)
Y.get_piecewise_cdf().plot(color='b', linewidth = 2.0) 

#! Heavy-tailed distribution
#! -------------------------

#$ Loss of the accuracy in integration procedure at $+\infty$. 
P = ParetoDistr(0.1)
figure()
demo_distr(P, xmin = 0, xmax = 1e2, ymax = 0.01)
figure()
demo_distr(P+P+P, xmin = 0, xmax = 1e2, ymax = 0.001)
#$ for higher exponents accuracy makes better 
P = ParetoDistr(0.25)
figure()
demo_distr(P, xmin = 0, xmax = 1e2, ymax = 0.01)
figure()
demo_distr(P + P + P, xmin = 0, xmax = 1e2, ymax = 0.01)
#! 
P = ParetoDistr(1.25)
figure()
demo_distr(P, xmin = 0, xmax = 1e2, ymax = 0.01)
figure()
demo_distr(P + P + P, xmin = 0, xmax = 1e3, ymax = 0.01)
#! One can try improve accuracz using changed integration parameters:
params.integration_infinite.exponent = 12
P = ParetoDistr(0.1)
figure()
demo_distr(P, xmin = 0, xmax = 1e2, ymax = 0.01)
#! 
P = ParetoDistr(0.25)
figure()
demo_distr(P, xmin = 0, xmax = 1e2, ymax = 0.01)

#! here we have extremally heavy tails, special integration procedures are needed
C = CauchyDistr()
figure()
demo_distr(exp(C))

#! Singularities outside zero
#! ---------------------------

#! Our interpolators give high accuracy around zero, however singularities 
#! outside zero are difficult to detection and such functions are difficult
#! to interpolation (see chebfun paper for details).
F = FunDistr(fun = lambda x:0.5/x**0.5, breakPoints=[0,0.5,1], lpoles = [True, False, False])
F.summary()
s = F + F
figure()
demo_distr(s, hist_points = 1000)
#$ For this case singularity at point 0 of $F$ is not such problem, more problematic is 
#$ dicontinuity at 1. 
#$ Loss of the accuracy is found on right nieghbouhood of point 1 and it is caused 
#$ by interpolator (absolute accuracy 1e-3, integral accuracy 1e-8).
#$ Here we try to interpolate function with unboded 1-st derivative.
#$ Other places seems to be evaluated correctly, e.g.:
print s(array(linspace(0,1,11))) - pi/4

#! To improve accuracy one can try change interpolation parameters (params.segments.maxn).
  
