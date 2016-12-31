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

n = 3
m = 3
X = []
E = []
Y = []
#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
t0 = time.time()

#Ei = MollifierDistr(0.4)
#Ei.summary()
#Ei.plot()
#show()

#A = UniformDistr(0,1, sym = "A")
#B = UniformDistr(0,1, sym = "B")
W = [BetaDistr(3,3, sym = "W" + str(i)) for i in range(m + 1)]
for i in range(n):
    #X.append(UniformDistr(0, 1, sym = "X{}".format(i)))
    xind = len(X)
    Yi = W[0]
    for j in range(m):
        X.append(BetaDistr(3, 3, sym = "X{}{}".format(i,j)))
        Yi += W[j+1] * X[-1]
    E.append(MollifierDistr(0.4, sym = "E{}".format(i)))
    Y.append(Yi + E[-1])
    Y[-1].setSym("Y{}".format(i))

M = Model(X + E + W, Y)
#M.eliminate_other(X + E + [A, B] + Y)
#M.toGraphwiz()


Xobs = []
Yobs = []
trueW = [0.3, 0.9, 0.5, 0.6]
k = 0
for i in range(n):
    yi = trueW[0]
    for j in range(m):
        Xobs.append(X[k].rand())
        k += 1
        yi += trueW[j+1] * Xobs[-1]
    yi += E[i].rand()
    Yobs.append(yi)


# #M.eliminate_other(X + Y)
# MXY = M.inference([X[0], Y[0]], [B], [0.7])
# figure()
# MXY.plot()
# plot([0.0, 1.0], [0.7, 0.7+a], "k-", linewidth=2.0)
# plot(Xobs, Yobs, "ko")
# plot()
# show()

#print ar, br
print(Xobs, Yobs)
print(X + Y)
print(concatenate((Xobs, Yobs)))

#print M
#MAB = M.inference([A,B]+E, X + Y,  concatenate((Xobs, Yobs)))
#print MAB
#MAB = MAB.inference([A,B], X + Y,  concatenate((Xobs, Yobs)))
MW = M.inference(W, X + Y,  concatenate((Xobs, Yobs)))
print("-------------------")
MW0 = MW.inference([W[0]],[],[])
#MB = MW.inference([B],[],[])

#M = M.inference([A,B], [X[0], Y[0]], [0.2, 0.4])
print(MW)
        
print(MW0)
figure()
subplot(211)
MW0.plot()
#print MB
#subplot(212)
#MB.plot()
#print "mean   est. A=", MA.as1DDistr().mean(),   "est. B=", MB.as1DDistr().mean()
#print "median est. A=", MA.as1DDistr().median(), "est. B=", MB.as1DDistr().median()
#print "mode   est. A=", MA.as1DDistr().mode(),   "est. B=", MB.as1DDistr().mode()
#MAB.nddistr(1, 3)

#paropt=fmin(lambda x,y: -MAB.nddistr(x,y), [MA.as1DDistr().mode(), MB.as1DDistr().mode()])
#print paropt
#print MAB.nddistr(paropt)

print(time.time() - t0)
show()
