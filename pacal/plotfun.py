from pylab import *
from numpy.lib.function_base import linspace, histogram
from matplotlib.pyplot import plot




        
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

