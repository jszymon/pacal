#!===================================================
#! The Distribution of Linear Regression Coefficients
#!===================================================

from numpy import concatenate, polyfit

from pylab import figure, show, plot, subplot, title


from pacal import *
from pacal.depvars.copulas import *
from pacal.depvars.models import Model
import time
from pacal.utils import maxprob

from scipy.optimize import fmin

from numpy.random import seed
seed(1)

params.interpolation_nd.maxn = 4
params.interpolation.maxn = 20
params.interpolation_pole.maxn = 20
params.interpolation_nd.debug_info = False
#$
#$ $y_i=ax_i+e_i,\quad, i=0,\ldots,\n-1$
#$
n = 10
X = []
E = []
Y = []
t0 = time.time()
A = UniformDistr(0,1, sym = "A")
B = UniformDistr(0,1, sym = "B")
for i in range(n):
    X.append(UniformDistr(0, 1, sym = "X{0}".format(i)))
    E.append(NormalDistr(0,0.2) | Between(-2,2))
    E[i].setSym("E{0}".format(i)) 
    Y.append(A * X[i]  + B + E[i])
    Y[i].setSym("Y{0}".format(i))

M = Model(X + E + [A, B], Y)
M.toGraphwiz(f=open('tmp.dot', mode="w+"))
#!
#! Sample
#!
Xobs = X[0].rand(n)
a, b = 0.3, 0.7 
Yobs = a * Xobs + b + E[0].rand(n)
print '{0:{align}15}\t{0:{align}15}'.format("Xobs","Xobs", align = '>')
for i in range(len(Xobs)):
    print '{0:{align}20}\t{0:{align}14}'.format(Xobs[i], Yobs[i], align = '>')
#!
#! Model
#!
print M
#!
#! Inference
#!
MAB = M.inference([A,B], X + Y,  concatenate((Xobs, Yobs)))
MA = MAB.inference([A],[],[])
MB = MAB.inference([B],[],[])
print MAB
print MA
print MB
figure()
MAB.plot(have_3d=True)
title("Joint distribution of A and B conditioned on sample")
#show()
figure()
MAB.plot(have_3d=False, cont_levels=10)
title("Joint distribution of A and B conditioned on sample")   
#show()
figure()
subplot(211)
MA.plot()
title("Marginalized distribution of A")
subplot(212)
MB.plot()
title("Marginalized distribution of B")
#show()
#!
#! Estimation coefficients
#!
print "original coefficients a=", a, "b=", b
print "mean   est. A=", MA.as1DDistr().mean(),   "est. B=", MB.as1DDistr().mean()
print "median est. A=", MA.as1DDistr().median(), "est. B=", MB.as1DDistr().median()
print "mode   est. A=", MA.as1DDistr().mode(),   "est. B=", MB.as1DDistr().mode()
MAB.nddistr(1, 3)
abopt = MAB.nddistr.mode()#maxprob(MAB.nddistr, array([0.5, 0.5]))
(ar,br)=polyfit(Xobs,Yobs,1)
print "              MAP =", abopt[0], abopt[1]
print "polyfit (LSE) est =", ar, br
print "time of doing=", time.time() - t0
show()


