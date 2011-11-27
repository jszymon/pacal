"""Simple differential equation."""

from pacal import *
from pylab import figure, show

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr


# y' = ay  Euler's method


#A = BetaDistr(1, 1, sym="A")
#A = UniformDistr(0.5, 1.75, sym="A")
A = UniformDistr(-1, 3, sym="A")
Y0 = BetaDistr(2, 6, sym="Y0")
n = 4
h = 1.0/n


K = (1 + h*A)
K.setSym("K") 
Y = [Y0]
for i in xrange(n):
    Y.append(Y[i] * K)
    Y[i+1].setSym("Y" + str(i+1))  
E = [BetaDistr(3,4,sym="E"+str(i)) for i in xrange(n+1)]
O = [Y[i] + E[i] for i in xrange(n+1)]
for i, o in enumerate(O):
    o.setSym("O"+str(i))
P = NDProductDistr([A, Y[0]] + E)
M = Model(P, O)
M.eliminate_other([K] + E + O + Y)
print M


#M.inference2(wanted_rvs = [A, Y[-1]], cond_rvs = [O[-1]], cond_X = [1.1])
#M.inference2(wanted_rvs = [A, Y[0]], cond_rvs = [O[-1]], cond_X = [1.1])
M2 = M.inference(wanted_rvs = [A, Y[0]], cond_rvs = [O[-1]], cond_X = [1.5])

print M2
M2.plot()
show()
import sys
sys.exit(0)

# or... do the elimination by hand
M.varschange(A, K)
print M
print M.nddistr
for i in xrange(n):
    M.varschange(E[i], O[i])
    M.eliminate(E[i])
    M.varschange(Y[i], Y[i+1])
    M.eliminate(Y[i])
    print M.nddistr
M.varschange(E[-1], O[-1])
M.eliminate(E[-1])
print M
print M.nddistr
#M.eliminate(O[-1])
#print M
#print M.nddistr
for i in xrange(n):
    M.eliminate(O[i])
    print M.nddistr
M.condition(O[-1], 1.1)
M.varschange(K, A)
M.eliminate(K)
M.plot()
#M.eliminate(Y[-1])
M.eliminate(A)
#M.condition(A, 0.8)
figure()
M.plot()
show()
0/0

show()
#M.varschange(X3, S3)
#print M
#M.condition(S3, 0.8)
#M.varschange(S2, X3)
#M.varschange(X3, X1)
