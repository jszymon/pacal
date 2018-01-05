#!===================================================
#! The Distribution of Linear Regression Coefficients
#!===================================================
from __future__ import print_function

from pacal import *

from pylab import figure, show, plot, subplot, title
from scipy.optimize import fmin
from numpy.random import seed
from numpy import concatenate, polyfit
seed(1)
import time
t0 = time.time()
#params.interpolation_nd.maxq = 7
params.interpolation.maxn = 20
params.interpolation_pole.maxn = 20
params.interpolation_nd.debug_info = False
#!
#! Sample
#!
n=10
Xobs = UniformDistr().rand(n)
a, b = 0.3, 0.7 
Yobs = a * Xobs + b #+ E[0].rand(n)
print('{0:{align}15}\t{0:{align}15}'.format("Xobs","Xobs", align = '>'))
for i in range(len(Xobs)):
    print('{0:{align}20}\t{0:{align}14}'.format(Xobs[i], Yobs[i], align = '>'))
#!
#! Model
#!
X = []
E = []
Y = []
A = UniformDistr(0,1, sym = "A")
B = UniformDistr(0,1, sym = "B")
for i in range(n):
    X.append(UniformDistr(0, 1, sym = "X{0}".format(i)))
    E.append(NormalDistr(0.0,0.2) | Between(-1,1))
    E[i].setSym("E{0}".format(i)) 
    Y.append(A * X[i]  + B + E[i])
    Y[i].setSym("Y{0}".format(i))

M = Model(X + E + [A, B], Y)
M.toGraphwiz(f=open('bn.dot', mode="w+"))
print(M)
#!
#! Inference
#!
MAB = M.inference([A,B], X + Y,  concatenate((Xobs, Yobs)))
MA = MAB.inference([A],[],[])
MB = MAB.inference([B],[],[])
print(MAB)
print(MA)
print(MB)
figure(1)
MAB.plot(have_3d=True)
title("Joint distribution of A and B conditioned on sample")
show()
figure(2)
MAB.plot(have_3d=False, cont_levels=10)
title("Joint distribution of A and B conditioned on sample")   
show()
figure(3)
subplot(211)
MA.plot()
title("Marginalized distribution of A")
subplot(212)
MB.plot()
title("Marginalized distribution of B")
show()
#!
#! Estimation of coefficients
#!
print("original coefficients a=", a, "b=", b)
print("mean   est. A=", MA.as1DDistr().mean(),   "est. B=", MB.as1DDistr().mean())
print("median est. A=", MA.as1DDistr().median(), "est. B=", MB.as1DDistr().median())
print("mode   est. A=", MA.as1DDistr().mode(),   "est. B=", MB.as1DDistr().mode())
MAB.nddistr(1, 3)
abopt = MAB.nddistr.mode()
(ar,br)=polyfit(Xobs,Yobs,1)
#!
#! Least squares estimates are maximum log-likelihood estimates 
#! (are equal to maximum a posteriori in this case, because of uniform a priori distribution of coefficients) 
#!
print("              MAP  a=", abopt[0], "b=", abopt[1])
print("polyfit (LSE) est. a=", ar, "b=", br)
print("computation time=", time.time() - t0)

