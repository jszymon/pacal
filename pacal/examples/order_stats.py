"""Demo of order statistics."""

from __future__ import print_function

from pacal import *
from pacal.utils import ordinal_ending
from pylab import figure, legend, title, xlim, ylim

if __name__ == "__main__":
    colors = "kbgrcmy"

    c = 0
    for i in range(1, 50, 5):
        os = iid_order_stat(BetaDistr(3, 2), 50, i)
        os.summary()
        os.plot(label = str(i) + "-" + ordinal_ending(i) + " ord. stat.", color = colors[c%len(colors)])
        xlim(0, 1)
        c += 1
    ylim(ymin = 0)
    legend(loc = "upper left")
    title("Order statistics from a sample of 50 Beta(3,2) r.v.'s")

    figure()
    N = 25
    title(str(N) + " independent Normal(0,1) r.v.'s")
    med = iid_median(NormalDistr(), N)
    med.summary()
    med.plot(label = "median", color = colors[0])
    m = iid_min(NormalDistr(), N)
    m.summary()
    m.plot(label = "minimum", color = colors[1])
    M = iid_max(NormalDistr(), N)
    M.summary()
    M.plot(label = "maximum", color = colors[2])
    os6 = iid_order_stat(NormalDistr(), N, 6)
    os6.summary()
    os6.plot(label = "6-th ord. stat", color = colors[3])
    os20 = iid_order_stat(NormalDistr(), N, 20)
    os20.summary()
    os20.plot(label = "20-th ord. stat", color = colors[4])
    legend()

    show()
