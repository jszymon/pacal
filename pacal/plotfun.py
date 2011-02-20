from pylab import *
from numpy.lib.function_base import linspace, histogram
from matplotlib.pyplot import plot




def plotdistr(d, l = -10, u = 10, numberOfPoints = 1000):
    X = linspace(l, u, numberOfPoints)
    #Y = [d.pdf(x) for x in X] # it should be vectorized
    Y = d.pdf(X) # doesn't work yet 
    plot(X,Y)
    
def histdistr(d, n = 1000000, l = None, u = None, bins = 50):
    if l is None and u is None:
        X = d.rand(n, None)
        allDrawn = len(X)
    else:
        X = []
        allDrawn = 0
        while len(X) < n:
            x = d.rand(n - len(X))
            allDrawn = allDrawn + len(x)
            if l is not None:
                x = x[(l <= x)]
            if u is not None:
                x = x[(x <= u)]
            X = hstack([X, x])
    dw = (X.max() - X.min()) / bins
    w = (float(n)/float(allDrawn)) / n / dw
    counts, binx = histogram(X, bins)
    width = binx[1] - binx[0]
    for c, b in zip(counts, binx):
        bar(b, float(c) * w, width = width, alpha = 0.25)
        
def dispNet(distr, file='', qm='', tab=''):
    file += tab
    file += distr.__str__()
    numberOfParents = len(distr.parents);
    #if numberOfParents == 0:
    #    tab += '..'
    if numberOfParents == 1:
        #file += '(\n'
        file += '\n'
        file=dispNet(distr.parents[0],file, '', tab+'   ')
        #file += tab + ')'
        file += ''    
    if numberOfParents == 2:
        file += '\n'
        #file += '(\n'
        file=dispNet(distr.parents[0],file, '', tab+'   ')
        #file=dispNet(distr.parents[0],file, ',', tab+'..')
        file += '\n'
        file=dispNet(distr.parents[1],file, '', tab+'   ')
        #file += '\n' + tab +')\n'
        file += '' 
    file += qm    
        
    #print file;
    return file;

