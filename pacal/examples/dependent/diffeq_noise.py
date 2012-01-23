#! ============================
#! Simple differential equation
#! ============================
from pacal import *
from pylab import figure, show, subplot, title, plot, xlabel, ylabel

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr
from numpy import array, zeros
import time
t0 = time.time()
params.interpolation_nd.maxq = 6
params.interpolation.maxn = 10
params.interpolation_pole.maxn =10
params.interpolation_nd.debug_info = True
params.models.debug_info = True


#!
#! Euler's method applied to equation y' = ay, with noisy observations 
#!
#! Y(i+1) = A * Y(i) 
#!
#! O(i) = Y(i+1) + E(i), i=0,...,n-1
#!

A = BetaDistr(2, 2, sym="A") # parameter of equation
Y0 = BetaDistr(2, 2, sym="Y0")  # initial value
#Y0 = UniformDistr(0, 1, sym="Y0")  # initial value
n = 10                           # number time points
h = 1.0 / n
K = (1 + h * A)
#K =UniformDistr(0.4, 0.9, sym="K")
K.setSym("K") 
Y, O, E, U  = [], [], [], []           # lists of states, observations and errors
for i in xrange(n):
    #Y.append(Y[i] * K)
    #Y[i + 1].setSym("Y" + str(i + 1))  
    U.append(UniformDistr(-0.1,0.1, sym="U{0}".format(i)))
    if i==0:
        Y.append(Y0 * K+ h*U[i])
    else:
        Y.append(Y[i-1] * K+ h*U[i])
    Y[-1].setSym("Y" + str(i+1))  
    ei = NormalDistr(0.1, 0.1) | Between(-0.3, 0.3)
    ei.setSym("E{0}".format(i))
    E.append(ei)
    O.append(Y[-1] + E[-1])
    O[-1].setSym("O{0}".format(i))
#! 
#! Model
#! -----
P = NDProductDistr([A, Y0] + E + U)
M = Model(P, O)
print M
M.eliminate_other(E + Y + O + [A, Y0] + U)
print M
M.toGraphwiz(f=open('bn.dot', mode="w+"))

#!
#! Joint distribution of initial condition and parameter of equation
#! -----------------------------------------------------------------
figure()
i = 0
ay0 = []
ui = [0.0]*(n/2) + [0.0]*(n/2) 
#ui = [0.1]*n
for yend in [0.2, 1.5]:
    M2 = M.inference(wanted_rvs=[A, Y0], cond_rvs=[O[-1]] + U, cond_X=[yend] + ui)
    subplot(1, 2, i + 1)
    title("given that observation O[{0}]={1}".format(n-1, yend))
    M2.plot()
    ay0.append(M2.nddistr.mode())           # "most probable" state
    print "yend=", yend, ",  MAP  est. of A, Y0 =", ay0[i]
    i += 1
#show()
#!
#! Trajectory
#! ----------
figure()

for j in range(len(ay0)):
    ymean, ystd = [], []
    subplot(1, 2, j + 1)
    for i in range(n):
        Myi = M.inference([O[i]], [A, Y0] + U, list(ay0[j]) + ui).as1DDistr()
                          
        ymean.append(Myi.mean())
        ystd.append(Myi.std())
        Myi.boxplot(pos=i+1, useci=0.05, showMean=False)
        
    title("A, Y[0] = {0:.3f},{1:.3f}".format(*ay0[j]))
    #plot(range(0, n, 1), ymean, 'k')
    #plot(range(0, n, 1), array(ymean) + ystd, 'k--')
    #plot(range(0, n, 1), array(ymean) - ystd, 'k--')
    ylabel("O[i]")
    xlabel("i")
show()
print "computation time=", time.time() - t0


