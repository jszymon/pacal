from pacal import *
from pylab import figure, show

from pacal.depvars.copulas import *
from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr

n = 10
X = list([0]*n)

Y = list([0]*n)

A = BetaDistr(3, 3, sym = "A")
B = BetaDistr(3, 3, sym = "B")
C = BetaDistr(2, 6, sym = "C")
D = BetaDistr(2, 6, sym = "D")
#A = 0.8 
#B = 0.2
#C = 0.1
#D = 0.9
h=0.1
for i in range(n):
    print i
    if i==0:
        X[i] = BetaDistr(3, 3, sym = "X0")
        Y[i] = BetaDistr(3, 3, sym = "Y0")
    else:
        X[i] = X[i-1] + h*(A*X[i-1] - B*Y[i-1])
        Y[i] = Y[i-1] + h*(-C*X[i-1] + D*Y[i-1])
        
        X[i].setSym("X{}".format(i))
        Y[i].setSym("Y{}".format(i))



M = Model([X[0], Y[0], A, B, C, D], X[1:] + Y[1:] )
print M
M.eliminate_other([X[0], Y[0], A, B, C, D] + X[1:] + Y[1:])
print M
print Y[1].range()

#M1 = M.inference([X[2], Y[2]], [X[0], Y[0]], [0.5, 0.2])\
figure()
    
for i in range(n):
    M1 = M.inference([X[i], Y[i]], [A, B, C, D], [0.9, 0.2, 0.3, 0.6])
    print M1
    M1.plot(cont_levels=1)
    #figure()
    #M1.plot(have_3d=True)
show()