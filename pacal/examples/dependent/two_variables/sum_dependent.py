"""Sum of two dependent variables."""

from __future__ import print_function

from pylab import legend, figure, title
from pacal import *

X = UniformDistr(1, 2, sym="X")
Y = BetaDistr(2, 2, sym="Y")
U = X + Y
U.setSym("U")

colors = "kbgrcmy"

title("Bivariate normal")
for i, theta in enumerate([-0.9, -0.5, 0.5, 0.9]):
    print("theta=", theta)
    ci = NDNormalDistr([0, 0], [[1, theta],[theta, 1]])
    Mi = TwoVarsModel(ci, ci.Vars[0] + ci.Vars[1])
    funi = Mi.eval()
    funi.plot(label = "theta={0}".format(theta), color = colors[i])
    funi.summary()
legend()
show()
figure()
title("GumbelCopula")
for i, theta in enumerate([1, 5, 10, 15]):
    print("theta=", theta)
    ci = GumbelCopula(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.plot(label = "theta={0}".format(theta), color = colors[i])
    funi.summary()
legend()
show()
figure()
title("FrankCopula")
for i, theta in enumerate([-15, -5, 5, 15]):
    print("theta=", theta)
    ci = FrankCopula(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.plot(label = "theta={0}".format(theta), color = colors[i])
    funi.summary()
legend()
show()
figure()
title("ClaytonCopula")
for i, theta in enumerate([1, 5, 10, 15]):
    print("theta=", theta)
    ci = ClaytonCopula(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.plot(label = "theta={0}".format(theta), color = colors[i])
    funi.summary()
legend()
show()
figure()
Y2 = NormalDistr(sym="Y2")
U2 = X + Y2
U2.setSym("U2")
title("FrankCopula - infinite domain")
for i, theta in enumerate([1, 5, 10, 15]):
    print("theta=", theta)
    ci = FrankCopula(marginals=[X, Y2], theta=theta)
    Mi = TwoVarsModel(ci, U2)
    funi = Mi.eval()
    funi.plot(label = "theta={0}".format(theta), color = colors[i])
    funi.summary()
legend()

show()
