from pacal import *
from pacal.depvars.copulas import *
from pacal.depvars.models import *

# ==== probability BMI ===============================



## ==== probability BMI ===============================
#X = BetaDistr(4, 3, sym="x1")
#Y = BetaDistr(3, 5, sym="x2")
##Y = UniformDistr(50, 120, sym="x1")
##X = UniformDistr(1.5, 2.1, sym="x2")
#
#cbmi = ClaytonCopula(marginals=[X, Y], theta=3)
#cbmi.plot()
#
#U = X/(X+Y) 
#M = TwoVarsModel(cbmi, U)
#M.plotFrame(20, 30, 3, 4)
#fun = M.varchange_and_eliminate()
#figure()
#
#X.plot()
#Y.plot()
#fun.plot()
#fun.summary()
#show()
#0/0
# ==== probability boxex ===============================
X = UniformDistr(1, 2, sym="x1")
Y = UniformDistr(1, 2, sym="x2")
cw = WCopula(marginals=[X, Y])
cw.plot()
cm = MCopula(marginals=[X, Y])
cm.plot()
#show()
cp = PiCopula(marginals=[X, Y])
U = X / Y #/ (Y + 1)# * X

Mw = TwoVarsModel(cw, U)
Mm = TwoVarsModel(cm ,U)
Mp = TwoVarsModel(cp ,U)
funw = Mw.varchange_and_eliminate()
funm = Mm.varchange_and_eliminate()
funp = Mp.varchange_and_eliminate()
figure()
funw.plot()
#funw.summary()
funm.plot()
funp.get_piecewise_cdf().plot()
funp.summary()
#for theta in [5, 10]:
#    print "::", theta
#    ci = GumbelCopula2d(marginals=[X, Y], theta=theta)
#    Mi = TwoVarsModel(ci, U)
#    funi = Mi.varchange_and_eliminate()
#    funi.get_piecewise_cdf().plot(color="g")
#    funi.summary()
#for theta in [-15, -5, 5, 15]:
#    print "::::", theta
#    ci = FrankCopula2d(marginals=[X, Y], theta=theta)
#    Mi = TwoVarsModel(ci, U)
#    funi = Mi.varchange_and_eliminate()
#    funi.get_piecewise_cdf().plot(color="b")
#    funi.summary()
#for theta in [5, 10]:
#    print ":::", theta
#    ci = ClaytonCopula(marginals=[X, Y], theta=theta)
#    Mi = TwoVarsModel(ci, U)
#    funi = Mi.varchange_and_eliminate()
#    funi.get_piecewise_cdf().plot(color="r")
#    funi.summary()
show()