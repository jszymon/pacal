from pacal import *

from pacal.depvars.models import Model
from pacal.depvars.copulas import FrankCopula2d, GumbelCopula2d, PiCopula

from pylab import figure

params.interpolation_nd.maxn = 7

X1 = BetaDistr(3,3, sym="X1")
X2 = BetaDistr(4,3, sym="X2")
Y1 = BetaDistr(3,7, sym="Y1")
Y2 = BetaDistr(5,4, sym="Y2")

X1 = BetaDistr(4,4, sym="X1")
X2 = BetaDistr(4,4, sym="X2")
Y1 = BetaDistr(4,4, sym="Y1")
Y2 = BetaDistr(4,4, sym="Y2")


C1 = FrankCopula2d(theta=5, marginals=[X1, X2])
C2 = FrankCopula2d(theta=2, marginals=[Y1, Y2])
C1.contour()
C2.contour()
figure()

# C1 = FrankCopula2d(theta=2, marginals=[X1, X2])
# C2 = FrankCopula2d(theta=2.5, marginals=[Y1, Y2])

#C1 = GumbelCopula2d(theta=2, marginals=[X1, X2])
#C1 = PiCopula(marginals=[X1, X2])


Z1 = X1 + Y1; Z1.setSym("Z1")
Z2 = X2 + Y2; Z2.setSym("Z2")

M = Model([C1,C2], [Z1, Z2])
#M = Model([C1,Y1,Y2], [Z1, Z2])
#M = Model([X1,X2,Y1,Y2], [Z1, Z2])
print M

M2 = M.inference([Z1, Z2])
print M2
M2.plot()
show()
