"""Simple differential equation."""

from pacal import *
from pylab import figure, show, zeros, plot, legend, subplot, rc

from matplotlib.lines import Line2D

rc('axes', labelsize=18)
rc('xtick', labelsize=15.0)
rc('ytick', labelsize=15.0)
rc('legend', fontsize=17.0)


linestyles = ["-", "--", "-.", ":"]
pfargs = {"linewidth":3, "color":"k", "dash_joinstyle":"round"}


from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr
from numpy.random import seed
seed(1)
from numpy import pi, std, array, concatenate, mean, abs
params.interpolation_nd.maxq = 2
params.interpolation.maxn = 100
params.interpolation_pole.maxn = 100
params.interpolation_nd.debug_info = False
params.interpolation.debug_info = False
params.models.debug_info = False
A = BetaDistr(3, 3, sym="A")    # parameter of equation
#Y0 = BetaDistr(3, 3, sym="Y0")  # initial value
Y0 = UniformDistr(-0.5, 0.5, sym="Y0")  # initial value
n = 3      # number time points
h = 1.0 / n
#A =-0.3
K = (1 - h * A)
#K = BetaDistr(5, 3, sym="A") #0.8
K0=0.7
K= K0
#K.setSym("K") 
Y = []                        # list of states
O, E, U = [], [], []            # lists of observations and errors
for i in xrange(n):
    print i
    U.append(UniformDistr(-0.2,0.2, sym="U{0}".format(i)))
    if i==0:
        Y.append(Y0 * K + U[i])
    else:
        Y.append(Y[i-1] * K + U[i])
    Y[i].setSym("Y" + str(i+1))  
    ei = NormalDistr(0.05, 0.1) | Between(-0.4, 0.4)
    ei.setSym("E{0}".format(i))
    #ei = MollifierDistr(0.5, sym="E{0}".format(i))
    E.append(ei)
    O.append(Y[-1] + E[-1])
    O[-1].setSym("O{0}".format(i))
    #print O[-1].range(), O[-1].range_()
M = Model(U+[Y0]+E, Y+O)
print M
M.toGraphwiz(f=open('bn.dot', mode="w+"))
nt = 100
u = zeros(nt)
t = zeros(nt)
Yorg = zeros(nt)
Ynoised = zeros(nt)
Ydenoised = zeros(nt)
Ydenoised1 = zeros(nt)
Ydenoised2 = zeros(nt)
Udenoised = zeros(nt)
yi = 0.0
ydenoise =  0.0
ynoise  = 0.0
y = 0.0
figure()
for i in range(nt):
    t[i] = i
    print "===========================================", i
    u[i] = 0.1*sign(sin(4 * pi * i / nt))
    #u[i] = 0.1*(sin(4 * pi * i / nt))
    y = y*K0 + u[i]
    Yorg[i] = y
    Ynoised[i] = y + E[0].rand()  
    
    if i>n-1:
        MY = M.inference([Y[-1]], O + U , concatenate((Ynoised[i-n+1:i+1], u[i-n+1:i+1])))
        ydenoised = MY.as1DDistr().mean()
        ydenoised1 = MY.as1DDistr().median()
        Ydenoised[i] = ydenoised
        Ydenoised1[i] = ydenoised1
        #MY.as1DDistr().boxplot(i, useci=0.05)
        
    print range(i-n+1,i+1,1)
    print i, Yorg[i], Ynoised[i], Ydenoised[i], Ydenoised1[i] 
#figure()
plot(t, u, 'k-', label="U", linewidth = 1.0)
plot(t, Ynoised, 'k.--', label="O", linewidth = 1.0)
plot(t, Yorg, 'k:', label="Y original", linewidth = 3.0)
#plot(t, Ydenoised, 'r-', label="X1 denoised")
plot(t, Ydenoised1, 'k-', label="Y denoised", linewidth = 2.0)
#plot(t, Ydenoised2, 'b-', label="X1 mode denoised")
#plot(t, Udenoised, 'b--', label="U1 denoised")
legend(loc='lower left')
print sqrt(mean((Yorg-Ynoised)**2)), sqrt(mean((Yorg-Ydenoised)**2)) 
print mean(abs(Yorg-Ynoised)), mean(abs(Yorg-Ydenoised)) 

print sqrt(mean((Yorg-Ynoised)**2)), sqrt(mean((Yorg-Ydenoised1)**2)) 
print mean(abs(Yorg-Ynoised)), mean(abs(Yorg-Ydenoised1)) 

show()