from pacal import *
from pylab import figure, show

from pacal.depvars.copulas import *
from pacal.depvars.models import Model
import numpy as _np
import time

n = 5
X = [0]*n
S = [0]*n
t0 = time.time()

#show()
A = UniformDistr(0,1, sym = "A")
B = UniformDistr(0,1, sym = "B")
for i in range(n):
    print "X{}".format(i)
    X[i] = BetaDistr(3, 3, sym = "X{}".format(i))
    if i==0:
        S[i] = X[0]        
    else:
        S[i] = S[i-1] + X[i]
    S[i].setSym("S{}".format(i))

M = Model(X, S[1:])
#M = M.inference([S[-1], S[-4]], [S[-3]], [1])
M = M.inference([X[0], X[1]], [S[-1]], [1])
print M
figure()
M.plot()
figure()
M.plot(have_3d=True)
show()
