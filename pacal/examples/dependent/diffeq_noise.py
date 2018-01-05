#! ============================
#! Simple differential equation
#! ============================
from __future__ import print_function

from pacal import *
from pylab import figure, show, subplot, title, plot, xlabel, ylabel, legend, rc

from matplotlib.lines import Line2D

rc('axes', labelsize=18)
rc('xtick', labelsize=15.0)
rc('ytick', labelsize=15.0)
rc('legend', fontsize=17.0)


linestyles = ["-", "--", "-.", ":"]
pfargs = {"linewidth":3, "color":"k", "dash_joinstyle":"round"}

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr
from numpy import array, zeros
import time
t0 = time.time()
params.interpolation_nd.maxq = 7
params.interpolation.maxn = 100
params.interpolation_pole.maxn =100
params.interpolation_nd.debug_info = False
params.models.debug_info = False


#!
#! Euler's method applied to equation y' = ay + u, with noisy observations 
#!
#! Y(i) = Y(i-1) + h * A * Y(i-1) + h * U[i-1]
#!
#! O(i) = Y(i) + E(i), i=1,...,n-1
#!

A = BetaDistr(3, 3, sym="A")    # parameter of equation
Y0 = BetaDistr(3, 3, sym="Y0")  # initial value
n = 10                          # number time points
h = 1.0 / n
K = (1 + h * A)
K.setSym("K") 
Y, O, E, U  = [], [], [], []           # lists of states, observations and errors
for i in range(n):
    #Y.append(Y[i] * K)
    #Y[i + 1].setSym("Y" + str(i + 1))  
    U.append(UniformDistr(-0.1,0.1, sym="U{0}".format(i)))
    # U will be conditioned on, so in effect constant
    if i==0:
        Y.append(Y0 * K+ h*U[i])
    else:
        Y.append(Y[i-1] * K+ h*U[i])
    Y[-1].setSym("Y" + str(i+1))  
    ei = NormalDistr(0.0, 0.1) | Between(-0.4, 0.4)
    ei.setSym("E{0}".format(i))
    E.append(ei)
    O.append(Y[-1] + E[-1])
    O[-1].setSym("O{0}".format(i))
#! 
#! Model
#! -----
P = NDProductDistr([A, Y0] + E + U)
M = Model(P, O)
print(M)
M.eliminate_other(E + Y + O + [A, Y0] + U)
print(M)
M.toGraphwiz(f=open('bn.dot', mode="w+"))

#!
#! Joint distribution of initial condition and parameter of equation
#! -----------------------------------------------------------------

i = 0
ay0 = []
ui = [0.0]*n
figure()
for yend in [0.25, 1.25, 2.25]:
    M2 = M.inference(wanted_rvs=[A, Y0], cond_rvs=[O[-1]] + U, cond_X=[yend] + ui)
    subplot(1, 3, i + 1)
    title("O_{0}={1}".format(n, yend))
    M2.plot()
    ay0.append(M2.nddistr.mode())           # "most probable" state
    print("yend=", yend, ",  MAP  est. of A, Y0 =", ay0[i])
    i += 1
show()
#!
#! Trajectory
#! ----------

figure()
styles=['-', '--', '-.', ':']
for j in range(len(ay0)):
    ymean, ystd = [], []
    for i in range(n):
        Myi = M.inference([O[i]], [A, Y0] + U, list(ay0[j]) + ui).as1DDistr()                   
        ymean.append(Myi.mean())
        ystd.append(Myi.std())
        Myi.boxplot(pos=i+1, useci=0.05, showMean=False)
    plot(list(range(1, n+1, 1)), ymean, 'k', linestyle=styles[j], label="A, Y[0] = {0:.3f},{1:.3f}".format(*ay0[j]))
    ylabel("O[i]")
    xlabel("i")
legend(loc='upper left')
show()
print("computation time=", time.time() - t0)


