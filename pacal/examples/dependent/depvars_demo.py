#!
#! ==============================================
#! Capabilities of PaCAL with dependent variables
#! ==============================================
#!
from __future__ import print_function

import sys; 
print(('%s %s' % (sys.executable or sys.platform, sys.version)))

from functools import partial

from pacal.distr import demo_distr
from pacal.depvars.nddistr import *
from pacal.depvars.copulas import *
from pacal.depvars.models import *
from pacal.segments import *
from pacal import *
 
from pylab import figure, show, rc, ylim, title, legend, subplot

params.interpolation.maxn = 100
params.interpolation.use_cheb_2nd = False

#!------------
#! Resistances
#!------------
R1 = UniformDistr(0.5, 1.5, sym="R1")
R2 = UniformDistr(1.5, 2.5, sym="R2")
pr = PiCopula(marginals=[R1, R2])
M = TwoVarsModel(pr, R1 * R2 / (R1 + R2))
R_total = M.eval()
R_total.plot(linewidth=2.0, color='k')
xlabel("R_total")
R_total.hist()
R_total.summary()

print(R_total.cdf(3.0 / 4.0))
print(R_total.median())

ylim(ymin=0)
#axis((0.3, 1.0, 0.0, 3.0))
show()


#! --------------------------------------------------------------------
#! Order statistics, sample range and interquantile range distribution
#! --------------------------------------------------------------------

figure()
X = NormalDistr(0, 1, sym="X")
CIJ = IJthOrderStatsNDDistr(X, n=8, i=2, j=7)
X2, X7 = CIJ.marginals 
X2.setSym("X2")
X7.setSym("X7")
M = TwoVarsModel(CIJ, X7 - X2)
R = M.varchange_and_eliminate()
R.plot(color="k", linewidth=2.0, linestyle="-", label=r'$X_{(7)} - X_{(2)}$ range')

X2.plot(color="k", linewidth=2.0, linestyle="--", label=r'$X_{(2)}$')
X7.plot(color="k", linewidth=2.0, linestyle=":", label=r'$X_{(7)}$')
R.summary()
axis((-6, 6.0, 0.0, 1.0))
legend()
show()
figure()
plot_2d_distr(CIJ)
show()
print(X7.mean() - X2.mean())
print(repr(R.mean()))


#! Sample range distribution
#! -------------------------
CIJ = IJthOrderStatsNDDistr(X, n=5, i=1, j=5)
X1, X5 = CIJ.marginals 
X1.setSym("X1")
X5.setSym("X5")

M = TwoVarsModel(CIJ, X5 - X1)
R = M.varchange_and_eliminate()
figure()
R.plot(color="k", linewidth=2.0, linestyle="-", label=r'$X_{(5)} - X_{(1)}$ range')
X1.plot(color="k", linewidth=2.0, linestyle="--", label=r'$X_{(1)}$')
X5.plot(color="k", linewidth=2.0, linestyle=":", label=r'$X_{(5)}$')
show()

figure()
plot_2d_distr(CIJ)
show()
#! Compare it with SAS/QC manual, functions: d2, d3
print(repr(R.mean()))
print(repr(R.var()), repr(R.std()))
R.summary()

#! ----------------------------------
#! Operations on correlated variables
#! ----------------------------------

X = UniformDistr(0, 1, sym="X")
Y = UniformDistr(0, 1, sym="Y")
B = X + Y
i = 0
for theta, ls in [(-15, "--"), (-5, "-."), (0.05, "-"), (10, ":")]:
    i += 1
    C = FrankCopula2d(theta=theta, marginals=[X, Y])
    M = TwoVarsModel(C, B)
    B_ = M.varchange_and_eliminate()
    print("---------------------------------------") 
    print(i, ", theta=", theta, ", tau=", C.tau(0, 1), ", rho=", C.rho_s(0, 1), ", corrcoef=",C.corrcoef(0, 1)) 
    figure(8)
    subplot(2, 2, i, projection='3d')
    C.plot(n=20, colors="k", labels=False)
    title(r'$\theta=$' + str(theta))

    figure(9)
    subplot(211)
    B_.plot(linewidth=2, linestyle=ls, color='k', label=r'$\theta=$' + str(theta))
    axis((0.0, 2.0, 0.0, 5.0))
    legend()
    subplot(212)
    B_.get_piecewise_cdf().plot(linewidth=2, linestyle=ls, color='k', label=r'$\theta=$' + str(theta))
    axis((0.0, 2.0, 0.0, 1.0))
    legend(loc=2)
figure(8)
show()
figure(9)
show()

#! -----------------------------
#! Regression, and median curves
#! -----------------------------

params.interpolation.maxn = 10
params.interpolation_pole.maxn = 10
params.interpolation.use_cheb_2nd = False

X, Y = UniformDistr(0, 1, sym="X1") + UniformDistr(0, 1, sym="X2"), BetaDistr(1, 4, sym="Y")# + UniformDistr()
X.setSym("X")
#F = ClaytonCopula(theta = 1.5, marginals=[X, Y])
#F = GumbelCopula(theta = 2, marginals=[X, Y])
figure()
F = FrankCopula(theta=4, marginals=[X, Y])
F.contour()
rx, ry = F.rand2d_invcdf(500)
plot(rx, ry, '.')

def regmean(x, F=F, type=2):
    distr = FunDistr(fun=lambda y: F.pdf(x, y) / X.pdf(x), breakPoints=Y.get_piecewise_pdf().getBreaks())
    if type == 1: return distr.mean()
    if type == 2: return distr.median()
    if type == 3: return distr.mode()     
    if type == 4: return distr.quantile(0.975)   
    if type == 5: return distr.quantile(0.025);     
xx = linspace(0.01, 1.99, 20)
cols = ["r", "g", "b", "k", "k"]
labels = ["mean", "median", "mode", "ci_U", "ci_L"]
for j in [1, 2, 3, 4, 5]:
    yy = zeros_like(xx)
    for i in range(len(xx)):
        yy[i] = regmean(xx[i], F, j)
    plot(xx, yy, cols[j - 1], linewidth=2.0, label=labels[j - 1])
legend(loc="upper left")
show()
