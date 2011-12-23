"""Simple differential equation."""

from pacal import *
from pylab import figure, show, zeros,plot

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr

from numpy import pi 
params.interpolation_nd.maxn = 3
params.interpolation.maxn = 20
params.interpolation_pole.maxn = 20
params.interpolation_nd.debug_info = True

# y' = ay  Euler's method


#A = BetaDistr(1, 1, sym="A")
#A = UniformDistr(0.5, 1.75, sym="A")
n = 1
A = NormalDistr(0.5, 0.2) | Between(0.1,0.9)
A.setSym("A")
A.parents=[]
#Y0 = BetaDistr(2, 2, sym="Y0")

Y = [UniformDistr(-10, 10, sym="Y0")]
U, E, O = [], [], []
for i in xrange(n):
    U.append(UniformDistr(-2,2,sym="U"+str(i)))
    Y.append(Y[i])
    Y[i+1].setSym("Y" + str(i+1))
    #ei= BetaDistr(3,3, sym="E{0}".format(i))
    ei = NormalDistr(0.4,0.6) | Between(-1,1)
    m = ei.mean()
    ei = NormalDistr(0.4,0.6)-m | Between(-1-m,1-m)
    ei.setSym("E{0}".format(i))
    E.append(ei)
    O.append(Y[i+1]*0.8 - E[i]+U[i])
    O[i].setSym("O{0}".format(i)) 
    #E = [BetaDistr(3,3,sym="E"+str(i)) for i in xrange(n+1)]
     
P = NDProductDistr([Y[0]] + E +U)
M = Model(P, O)
M.eliminate_other(E + Y+O+U+[A])
M.toGraphwiz()#f=open('bn.dot', mode="w+"))
print M
nt=30
u = zeros(nt)
t = zeros(nt)
Ymean, Ymode = [], []
Ymedian = []
Ynoise =[]
yi = 0.3
for i in range(nt):
    t[i]=i
    u[i]=sign(sin(4*pi*i/nt))
    print [0.3, 0.3, u[i]]
    #MYi = M.inference([O[0]], [Y[0], A, U[0]], [0.3, 0.3, u[i]]).as1ddistr()
    MYi = M.inference([O[0]],[Y[0], U[0]], [yi,u[i]]).as1DDistr()
    Ymean.append(MYi.mean()) 
    Ymode.append(MYi.mode()) 
    Ymedian.append(MYi.median())
    if len(Ynoise)==0:
        Ynoise.append(yi + u[i])
    else:
        Ynoise.append(Ynoise[-1]*0.8 + u[i])
    yi=Ymean[-1]
    print i, yi, yi,u[i],float(E[0].rand())
    
plot(t, u, 'k')
plot(t, Ymean,'b')
#plot(t, Ymode,'r')
plot(t, Ymedian,'g')
plot(t, Ynoise,'k--')
plot(t, -E[0].rand(nt)  + Ynoise,'r--')
show()