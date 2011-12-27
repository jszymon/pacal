"""Simple differential equation."""

from pacal import *
from pylab import figure, show, zeros, plot, legend

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr

from numpy import pi, std, array
params.interpolation_nd.maxn = 3
params.interpolation.maxn = 20
params.interpolation_pole.maxn = 20
params.interpolation_nd.debug_info = True

# y' = ay  Euler's method


#A = BetaDistr(1, 1, sym="A")
#A = UniformDistr(0.5, 1.75, sym="A")
n = 1
A = NormalDistr(0.5, 0.2) | Between(0.1, 0.9)
A.setSym("A")
A.parents = []
#Y0 = BetaDistr(2, 2, sym="Y0")

Y = [UniformDistr(-10, 10, sym="Y0")]
U, E, O = [], [], []
for i in xrange(n):
    U.append(UniformDistr(-2, 2, sym="U" + str(i)))
    Y.append(Y[i])
    Y[i + 1].setSym("Y" + str(i + 1))
    #ei= BetaDistr(3,3, sym="E{0}".format(i))
    ei = NormalDistr(0.0, 2.2) | Between(-1, 1)
    m = ei.mean()
    ei = NormalDistr(0.0, 2.2) - m | Between(-1 - m, 1 - m)
    ei.setSym("E{0}".format(i))
    E.append(ei)
    O.append(Y[i + 1] * A - E[i] + U[i])
    O[i].setSym("O{0}".format(i)) 
    #E = [BetaDistr(3,3,sym="E"+str(i)) for i in xrange(n+1)]
     
P = NDProductDistr([Y[0],A] + E + U)
M = Model(P, O)
M.eliminate_other(E + Y + O + U + [A])
M.toGraphwiz()#f=open('bn.dot', mode="w+"))
print M
nt = 100
u = zeros(nt)
t = zeros(nt)
Ymean, Ymode = [], []
Ymedian = []
Ynoise, Yorg = [], []
yi = 0.3
for i in range(nt):
    t[i] = i
    u[i] = sign(sin(2 * pi * i / nt))
    print [0.3, 0.3, u[i]]
    #MYi = M.inference([O[0]], [Y[0], A, U[0]], [0.3, 0.3, u[i]]).as1ddistr()
    MYi = M.inference([O[0]], [Y[0], U[0],A], [yi, u[i],0.5]).as1DDistr()
    Ymean.append(MYi.mean()) 
    Ymode.append(MYi.mode()) 
    Ymedian.append(MYi.median())
    if len(Ynoise) == 0:
        Yorg.append(yi + u[i])
        Ynoise.append(yi + u[i] + float(E[0].rand()))
    else:
        Yorg.append(Yorg[-1] * 0.6 + u[i])
        Ynoise.append(Ynoise[-1] * 0.6 + u[i] + float(E[0].rand()))
    yi = Ymean[-1]
    print i, yi, yi, u[i], float(E[0].rand())
    
plot(t, u, 'k', label="U")
plot(t, Ymean, 'r', label="Ymean")
#plot(t, Ymode,'r')
plot(t, Ymedian, 'g', label="Ymedian")
plot(t, Ynoise, 'k', label="Ynoise")
plot(t, Yorg, 'b', label="Yorg")
legend()
YnoiseK = E[0].rand(nt) + Ymean
plot(t, YnoiseK, 'r--')
print std(array(Yorg) - Ynoise)
print std(YnoiseK - Yorg)
print std(array(Ymean) - Yorg)
print std(array(Ymedian) - Yorg)
show()
