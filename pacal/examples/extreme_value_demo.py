"""Extreme value distribution demo."""

from pacal import *

EulConst = 0.5772156649015328606

colors = "kbgrcmy"
def extreme_limit_demo(X, N = 5, xmin = None, xmax = None, ymax = None, **args):
    figure()
    title("Limit of maxima of " + X.getName())
    X.plot(linewidth = 4, color = "c", **args)
    #Ys = iid_max(
    Y = X
    print "Limit of maxima of " + X.getName() + ": ",
    for i in xrange(N-1):
        print i+2,
        sys.stdout.flush()
        Y += X
        (Y/(i+2)).plot(color = colors[i%len(colors)], **args)
    if xmin is not None:
        xlim(xmin = xmin)
    if xmax is not None:
        xlim(xmax = xmax)
    ylim(ymin = 0)
    if ymax is not None:
        ylim(ymax = ymax)
    print
    #show()

X = ExponentialDistr()
#X = NormalDistr()
M = iid_max(X, 100)
M.plot(linewidth=4, color="m")
M.summary()
m = M.mean()
g = GumbelDistr(m-EulConst)
g.plot()

show()
