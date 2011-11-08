
from pacal.depvars.copulas import *
from pacal.depvars.models import *
from pacal import *
from pacal.segments import *

 
params.interpolation.maxn = 10
params.interpolation.use_cheb_2nd = False
                
# ==== probability BMI ===============================

from pacal.depvars.copulas import *
from pacal.depvars.models import *
from pacal import *
from pacal.segments import *

 
params.interpolation.maxn = 10
params.interpolation.use_cheb_2nd = False
    
X, Y = UniformDistr() + UniformDistr(), BetaDistr(1, 4) 
#c = ClaytonCopula(theta = 0.5, marginals=[X, Y])
F = FrankCopula(theta = 8, marginals=[X, Y])            
#cp = PiCopula(marginals=[X, Y])
#cp = ClaytonCopula(theta=2, marginals=[X, Y])
U = X  +Y
M = TwoVarsModel(F ,U)
y = M.varchange_and_eliminate()
t0 = time.time()
y.plot()
print time.time() -t0
t0 = time.time()
U.plot()
print time.time() -t0show()
show()
0/0
# ==== Copulas, regression =============================
#X, Y = BetaDistr(3, 6, sym="X"), BetaDistr(3, 1, sym="Y")
X, Y = BetaDistr(2,3), UniformDistr() + UniformDistr()
#c = ClaytonCopula(theta = 0.5, marginals=[X, Y])
F = FrankCopula(theta = 8, marginals=[X, Y])

def _fun(x, type=3):
    if isscalar(x):
        distr = FunDistr(fun=lambda y: F.pdf(x,y)/X.pdf(x), breakPoints=Y.get_piecewise_pdf().getBreaks())
        if type==1: return distr.mean()
        if type==2: return distr.median()
        if type==3: return distr.mode()        
    else:
        y =  zeros_like(x)
        for i in range(len(x)):
            y[i] = _fun(x[i])
        return y 
y = linspace(0,1,100)
z = F.pdf(0.1,y)/Y.pdf(y)

F.plot()
F.contour()
t0 = time.time()
print "a"
rx,ry = F.rand2d_invcdf(500)
print "b", time.time() - t0
plot(rx,ry,'.')

mreg = PiecewiseFunction(fun=_fun, breakPoints=X.get_piecewise_pdf().getBreaks()).toInterpolated()
mreg.plot()

figure()
distr = FunDistr(fun=lambda y: F.pdf(0.04,y)/X.pdf(0.04), breakPoints=Y.get_piecewise_pdf().getBreaks())
distr.summary()
distr.plot()
show()
0/0
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
#fun = M.eval()
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
Y = BetaDistr(2, 2, sym="x2")
cw = WCopula(marginals=[X, Y])
cw.plot()
cm = MCopula(marginals=[X, Y])
cm.plot()
figure()
#show()
cp = PiCopula(marginals=[X, Y])
U = Y + X #/ (Y + 1)# * X

Mw = TwoVarsModel(cw, U)
Mm = TwoVarsModel(cm ,U)
Mp = TwoVarsModel(cp ,U)
#funw = Mw.eval()
#funm = Mm.eval()
funp = Mp.eval()
figure()
#funw.plot()
#funw.summary()
#funm.plot()
funp.get_piecewise_cdf().plot(color="k", linewidth=2.0)
funp.summary()
for theta in [5, 10]:
    print "::", theta
    ci = GumbelCopula2d(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.get_piecewise_cdf().plot(color="g")
    funi.summary()
for theta in [-15, -5, 5, 15]:
    print "::::", theta
    ci = FrankCopula2d(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.get_piecewise_cdf().plot(color="b")
    funi.summary()
for theta in [5, 10]:
    print ":::", theta
    ci = ClaytonCopula(marginals=[X, Y], theta=theta)
    Mi = TwoVarsModel(ci, U)
    funi = Mi.eval()
    funi.get_piecewise_cdf().plot(color="r")
    funi.summary()
show()