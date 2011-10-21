from pacal import *
from pylab import figure, show

from pacal.depvars.copulas import *
from pacal.depvars.models import Model

n = 1
X = []
E = []
Y = []
A = UniformDistr(0,1, sym = "A")
B = UniformDistr(0,1, sym = "B")
for i in range(n):
    X.append(BetaDistr(3, 3, sym = "X{}".format(i)))
    E.append(MollifierDistr(0.3, sym = "E{}".format(i)))    
    Y.append(A * X[i]  + B + E[i])
    Y[i].setSym("Y{}".format(i))

M = Model(X + E + [A, B], Y)
M.eliminate_other(X + E + [A, B] + Y)

Xobs = X[0].rand(n)
Yobs = 0.5*Xobs+0.0

print Xobs, Yobs
print X + Y
print array([Xobs, Yobs])

print M
M = M.inference([A,B], X + Y, Xobs+Yobs)
#M = M.inference([A,B], [X[0], Y[0]], [0.2, 0.4])
print M
M.plot()
show()