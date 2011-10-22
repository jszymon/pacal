from pacal import *
from pylab import figure, show

from pacal.depvars.copulas import *
from pacal.depvars.models import Model
import numpy as _np
import time


n = 10
X = []
E = []
Y = []
#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
t0 = time.time()

Ei = MollifierDistr(0.4)
Ei.summary()
Ei.plot()
#show()
A = UniformDistr(0,1, sym = "A")
B = UniformDistr(0,1, sym = "B")
for i in range(n):
    X.append(BetaDistr(3, 3, sym = "X{}".format(i)))
    E.append(MollifierDistr(0.4, sym = "E{}".format(i)))    
    Y.append(A * X[i]  + B + E[i])
    Y[i].setSym("Y{}".format(i))

M = Model(X + E + [A, B], Y)
M.eliminate_other(X + E + [A, B] + Y)

Xobs = X[0].rand(n)
Yobs = 0.3*Xobs+0.7

print Xobs, Yobs
print X + Y
print _np.concatenate((Xobs, Yobs))

print M
MAB = M.inference([A,B]+E, X + Y,  _np.concatenate((Xobs, Yobs)))
print MAB
MAB = MAB.inference([A,B], X + Y,  _np.concatenate((Xobs, Yobs)))
MA = MAB.inference([A],[],[])
MB = MAB.inference([B],[],[])
print MA.nddistr
print MA.nddistr.__class__

#M = M.inference([A,B], [X[0], Y[0]], [0.2, 0.4])
print MAB
figure()
MAB.plot(have_3d=True)
figure()
MAB.plot(have_3d=False)
        
print MA
figure()
subplot(211)
MA.plot()
print MB
subplot(212)
MB.plot()
print "mean   est. A=", MA.as1DDistr().mean(),   "est. B=", MB.as1DDistr().mean()
print "median est. A=", MA.as1DDistr().median(), "est. B=", MB.as1DDistr().median()
print "mode   est. A=", MA.as1DDistr().mode(),   "est. B=", MB.as1DDistr().mode()
print time.time() - t0
show()