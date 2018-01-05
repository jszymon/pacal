from __future__ import print_function

from pacal.depvars.models import Model
from pacal import *

#from pacal.depvars.copulas import *
#from pacal.depvars.models import Model
from pylab import figure, show, rc
import numpy as _np
import time

params.interpolation_nd.maxq = 9
rc('axes', labelsize=18)
rc('xtick', labelsize=15.0)
rc('ytick', labelsize=15.0)
rc('legend', fontsize=17.0)


linestyles = ["-", "--", "-.", ":"]
pfargs = {"linewidth":3, "color":"k", "dash_joinstyle":"round"}
from numpy import ceil, isscalar, zeros_like, asfarray

n = 5
X = [0]*n
S = [0]*n
t0 = time.time()

#show()
#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
for i in range(n):
    print("X{}".format(i))
    X[i] = BetaDistr(2, 2, sym = "X{}".format(i))
    if i==0:
        S[i] = X[0]        
    else:
        S[i] = S[i-1] + X[i]
        S[i].setSym("S{}".format(i))

M = Model(X, S[1:])
print(M)
M.toGraphwiz()
#M = M.inference([S[-1], S[-4]], [S[-3]], [1])
#M = M.inference([X[0], X[1]], [S[-1]], [3.5])
print("====================")
M1 = M.inference(wanted_rvs =[X[0], X[1]], cond_rvs=[S[-1]], cond_X=[1])
print("====================",M1)
M2 = M.inference(wanted_rvs =[S[1], S[4]])
print("====================",M2)
M3 = M.inference(wanted_rvs =[S[1], S[4]], cond_rvs=[S[3]], cond_X=[2])
print("====================",M3)
MC_X0 = M.inference(wanted_rvs =[X[0]], cond_rvs=[S[-1]], cond_X=[1])
print("====================")

print(M1)
figure()
M1.plot(cont_levels=10)
figure()
M1.plot(have_3d=True)

print(M2)
figure()
M2.plot(cont_levels=10)
figure()
M2.plot(have_3d=True)

print(M3)
figure()
M3.plot(cont_levels=10)
figure()
M3.plot(have_3d=True)

print(MC_X0)
figure()
MC_X0.plot()

show()
