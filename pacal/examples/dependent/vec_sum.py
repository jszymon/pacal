from pacal import *

from pacal.depvars.models import Model

X1 = BetaDistr(3,3, sym="X1")
X2 = BetaDistr(4,3, sym="X2")
Y1 = BetaDistr(3,7, sym="Y1")
Y2 = BetaDistr(5,4, sym="Y2")

C1 = FrankCopula(theta=2, marginals=[X1, X2])
C2 = FrankCopula(theta=2.5, marginals=[Y1, Y2])

Z1 = X1 + Y1; Z1.setSym("Z1")
Z2 = X2 + Y2; Z2.setSym("Z2")

M = Model([C1,C2], [Z1, Z2])
print M

M.inference([Z1, Z2])
M.plot()
show()
