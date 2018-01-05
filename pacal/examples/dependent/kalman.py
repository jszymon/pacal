#! =================================
#! Kalman filter with control input
#! =================================
from __future__ import print_function

from pacal import *
from pylab import figure, show, zeros, plot, legend, subplot, rc

from matplotlib.lines import Line2D

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr
from numpy import pi, std, array, concatenate, mean, abs
from numpy.random import seed
seed(1)

rc('axes', labelsize=18)
rc('xtick', labelsize=15.0)
rc('ytick', labelsize=15.0)
rc('legend', fontsize=17.0)


linestyles = ["-", "--", "-.", ":"]
pfargs = {"linewidth":3, "color":"k", "dash_joinstyle":"round"}

params.interpolation_nd.maxq = 2
params.interpolation.maxn = 50
params.interpolation_pole.maxn = 50
params.interpolation_nd.debug_info = False
params.interpolation.debug_info = False
params.models.debug_info = True
#!
#! Model
#! ------
#!
#! Y(i) = K * Y(i-1) + U 
#!
#! O(i) = Y(i) + E(i), i=0,...,n-1
#!
A = BetaDistr(3, 3, sym="A")    # parameter of equation
Y0 = UniformDistr(-0.5, 0.5, sym="Y0")      # initial value
n = 3                           # number time points
h = 1.0 / n
K = 0.7
Y = []                          # list of states
O, E, U = [], [], []            # lists of observations and errors
for i in range(n):
    U.append(UniformDistr(-0.2, 0.2, sym="U{0}".format(i)))
    if i == 0:
        Y.append(Y0 * K + U[i])
    else:
        Y.append(Y[i - 1] * K + U[i])
    Y[i].setSym("Y" + str(i + 1))  
    ei = NormalDistr(0.05, 0.1) | Between(-0.4, 0.4)
    ei.setSym("E{0}".format(i))
    E.append(ei)
    O.append(Y[-1] + E[-1])
    O[-1].setSym("O{0}".format(i))
    #print O[-1].range(), O[-1].range_()
M = Model(U + [Y0] + E, Y + O)
print(M)
M.toGraphwiz(f=open('bn.dot', mode="w+"))
#!
#! Simulation with signal filtering
#! --------------------------------
nT = 100
u = zeros(nT)
t = zeros(nT)
Yorg = zeros(nT)
Ynoised = zeros(nT)
Ydenoised = zeros(nT)
Udenoised = zeros(nT)
yi = 0.0
ydenoise = 0.0
ynoise = 0.0
y = 0.0
figure()
for i in range(nT):
    t[i] = i
    # Deterministic simultation
    u[i] = 0.1 * sign(sin(4 * pi * i / nT))
    y = y * K + u[i]
    Yorg[i] = y
    Ynoised[i] = y + E[0].rand()  
    # Inference (Y[i] | O[i-n+1], ..., O[i]
    if i > n - 1:
        MY = M.inference(wanted_rvs=[Y[-1]], cond_rvs=O + U , cond_X=concatenate((Ynoised[i - n + 1:i + 1], u[i - n + 1:i + 1])))
        ydenoised = MY.as1DDistr().median()
        Ydenoised[i] = ydenoised
        #MY.as1DDistr().boxplot(i, width=0.2, useci=0.1)
plot(t, u, 'k-', label="U", linewidth=1.0)
plot(t, Ynoised, 'k.--', label="O", linewidth=1.0)
plot(t, Yorg, 'k-.', label="Y original", linewidth=3.0)
plot(t, Ydenoised, 'k-', label="Y denoised", linewidth=2.0)
legend(loc='lower left')

#! Error of estimation using median 
#! --------------------------------
print("mse=", sqrt(mean((Yorg - Ynoised) ** 2)), sqrt(mean((Yorg - Ydenoised) ** 2))) 
print("mae=", mean(abs(Yorg - Ynoised)), mean(abs(Yorg - Ydenoised))) 


show()
