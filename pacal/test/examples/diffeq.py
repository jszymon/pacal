"""Simple differential equation."""


from pylab import figure, show

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr

from  numpy import *
from pacal import *

from pylab import plot, semilogx, xlabel, ylabel, axis, loglog, figure, subplot, rc

import time
from matplotlib.lines import Line2D

rc('axes', labelsize=18)
rc('xtick', labelsize=15.0)
rc('ytick', labelsize=15.0)
rc('legend', fontsize=17.0)


linestyles = ["-", "--", "-.", ":"]
pfargs = {"linewidth":3, "color":"k", "dash_joinstyle":"round"}
from numpy import ceil, isscalar, zeros_like, asfarray

# y' = ay  Euler's method


#A = UniformDistr(0, 1, sym="A")
A = BetaDistr(3, 3, sym="A")
B = BetaDistr(2, 4, sym="A")
Y0 = BetaDistr(2, 2, sym="Y0")
n = 5
h = 1.0/n


K = (1 + h*A)
K.setSym("K") 
Y = [Y0]*(n+1)
for i in range(n+1):
    if i==0:
        pass
    else:
        Y[i] = Y[i-1] * K
        Y[i].setSym("Y" + str(i))  
P = NDProductDistr([Factor1DDistr(A), Factor1DDistr(Y[0])])
M = Model(P, Y[1:])
M.eliminate_other([K] + Y)

#M2 = M.inference2([Y[0], A], [Y[n]], [1])
#M2.plot(); print M2; show()
#M2 = M.inference2([Y[0]], [Y[n]], [0.5])
#figure()
#M2.plot(); print M2; 
figure()
Y[-1].plot(color='r',linewidth=5)
M3 = M.inference([Y[-1]], [], [])

M3.plot(); print(M3); 

X0 = BetaDistr(2, 2)
y = X0 * exp(A) 
y.summary()
y.plot(label="Y0*exp(A)")
Y[-1].plot('r')
figure()
err = y.get_piecewise_pdf() - M3.as1DDistr().get_piecewise_pdf()
err.plot()
show()

stop

print("---", [K] + Y)
print(M)
M.varschange(A, K)
print(M)
for i in range(n):
    M.varschange(Y[i], Y[i+1])
print(M)
M.varschange(K, A)
M.plot()
#M.condition(Y[n], 2)
print(M)
M.eliminate(K)
print(M)
for i in range(n-1,-1,-1):
    M.eliminate(Y[i])
print(M)
M.eliminate(A)
print(M)
figure()
M.plot()
print(M.nddistr.pdf(linspace(0,2.5,100)))
X0 = BetaDistr(1, 1)
y = X0 * exp(A)
y.summary()
y.plot(label="Y0*exp(A)")
figure()
err = y.get_piecewise_pdf() - M.as1DDistr().get_piecewise_pdf()
err.plot()

show()
#M.varschange(X3, S3)
#print M
#M.condition(S3, 0.8)
#M.varschange(S2, X3)
#M.varschange(X3, X1)
