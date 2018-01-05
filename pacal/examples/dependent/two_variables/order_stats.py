"""Joint distribution of two order statistics.

Distribution of sample range."""

from __future__ import print_function

from pylab import figure, legend, title

from pacal import *

X = NormalDistr(0, 1, sym="X")

CIJ = IJthOrderStatsNDDistr(X, n=5, i=1, j=5)
X1, X5 = CIJ.marginals 
X1.setSym("X1")
X5.setSym("X5")

M = TwoVarsModel(CIJ, X5 - X1)
R = M.eval()

R.plot(color="k", linewidth=2.0, linestyle="-", label=r'$X_{(5)} - X_{(1)}$'+'\n(sample range)')
X1.plot(color="k", linewidth=2.0, linestyle="--", label=r'$X_{(1)}$')
X5.plot(color="k", linewidth=2.0, linestyle=":", label=r'$X_{(5)}$')
legend()

figure()
CIJ.plot()
title("Joint distribution of min and max of 5 normals")

#! Compare it with SAS/QC manual, functions: d2, d3
print(repr(R.mean()))
print(repr(R.var()), repr(R.std()))
R.summary()

show()
