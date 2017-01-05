"""Parallel connection of two resistors."""

from __future__ import print_function

from pylab import figure, title, ylim
from pacal import *

def plot_parallel_resistors(R1, R2):
    N = R1 * R2
    D = R1 + R2
    R = N / D

    M = TwoVarsModel(PiCopula([R1, R2]), R)
    r = M.eval()
    r.plot()
    ylim(ymin = 0)
    r.summary()


R1 = UniformDistr(0.5, 1.5)
R2 = UniformDistr(1.5, 2.5)
plot_parallel_resistors(R1, R2)

title("R1~U(0.5, 1.5), R2~U(1.5, 2.5), R= (R1*R2) / (R1 + R2)")
show()

figure()
R = 1/(1/R1+1/R2)
R.plot()
R.summary()
ylim(ymin = 0)
title("1/(1/R1 + 1/R2)")
show()
