from numpy import concatenate, polyfit

from pylab import figure, show, plot, subplot


from pacal import *
from pacal.depvars.copulas import *
from pacal.depvars.models import Model
import time

from scipy.optimize import fmin

from numpy.random import seed
seed(1)

#params.interpolation_nd.maxn = 8

n = 20
X = []
E = []
Y = []
#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
t0 = time.time()

Ei = MollifierDistr(0.4)
Ei.summary()
#Ei.plot()
#show()

#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
A = BetaDistr(3,3, sym = "A")
B = BetaDistr(3,3, sym = "B")
for i in range(n):
    #X.append(UniformDistr(0, 1, sym = "X{}".format(i)))
    X.append(BetaDistr(3, 3, sym = "X{}".format(i)))
    E.append(MollifierDistr(0.4, sym = "E{}".format(i)))    
    Y.append(A * X[i]  + B + E[i])
    Y[i].setSym("Y{}".format(i))

M = Model(X + E + [A, B], Y)
#M.eliminate_other(X + E + [A, B] + Y)
M.toGraphwiz()



Xobs = X[0].rand(n)
a = 0.3 #A.rand(1)[0]
Yobs = a*Xobs+0.7 + Ei.rand(n)

b = 0.7
Yobs = a*Xobs+0.7 + Ei.rand(n)
print "a=", a
(ar,br)=polyfit(Xobs,Yobs,1)
#figure()
#plot(Xobs, Yobs, 'o')


# #M.eliminate_other(X + Y)
# MXY = M.inference([X[0], Y[0]], [B], [0.7])
# figure()
# MXY.plot()
# plot([0.0, 1.0], [0.7, 0.7+a], "k-", linewidth=2.0)
# plot(Xobs, Yobs, "ko")
# plot()
# show()

print ar, br
print Xobs, Yobs
print X + Y
print concatenate((Xobs, Yobs))

#print M
#MAB = M.inference([A,B]+E, X + Y,  concatenate((Xobs, Yobs)))
#print MAB
#MAB = MAB.inference([A,B], X + Y,  concatenate((Xobs, Yobs)))
MAB = M.inference([A,B], X + Y,  concatenate((Xobs, Yobs)))
MA = MAB.inference([A],[],[])
MB = MAB.inference([B],[],[])

#M = M.inference([A,B], [X[0], Y[0]], [0.2, 0.4])
print MAB
figure()
MAB.plot(have_3d=True)
figure()
MAB.plot(have_3d=False, cont_levels=10)
        
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
MAB.nddistr(1, 3)

#paropt=fmin(lambda x,y: -MAB.nddistr(x,y), [MA.as1DDistr().mode(), MB.as1DDistr().mode()])
#print paropt
#print MAB.nddistr(paropt)

print time.time() - t0
show()
