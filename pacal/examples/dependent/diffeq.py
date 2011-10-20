"""Simple differential equation."""


from pylab import figure, show

from pacal.depvars.models import Model
from pacal.depvars.nddistr import NDProductDistr, Factor1DDistr

from  numpy import *
from pacal import *
# y' = ay  Euler's method


A = UniformDistr(0, 1, sym="A")
Y0 = UniformDistr(0, 1, sym="Y0")
n = 5
h = 1.0/n


K = (1 + h*A)
K.setSym("K") 
Y = [Y0]
for i in xrange(n):
    Y.append(Y[i] * K)
    Y[i+1].setSym("Y" + str(i+1))  
P = NDProductDistr([Factor1DDistr(A), Factor1DDistr(Y[0])])
M = Model(P, Y[1:])
M.eliminate_other([K] + Y)

#M2 = M.inference2([Y[0], A], [Y[n]], [1])
#M2.plot(); print M2; show()
M2 = M.inference2([Y[0]], [Y[n]], [1])
figure()
M2.plot(); print M2; show()
stop

print "---", [K] + Y
print M
M.varschange(A, K)
print M
for i in xrange(n):
    M.varschange(Y[i], Y[i+1])
print M
M.varschange(K, A)
M.plot()
#M.condition(Y[n], 2)
print M
M.eliminate(K)
print M
for i in xrange(n-1,-1,-1):
    M.eliminate(Y[i])
print M
M.eliminate(A)
print M
figure()
M.plot()
print M.nddistr.pdf(linspace(0,2.5,100))
X0 = BetaDistr(1, 1)
y = X0 * exp(A)
y.summary()
y.plot(label="Y0*exp(A)")
figure()
err = y.get_piecewise_pdf() - M.as1DDistr().get_piecewise_pdf()
err.plot()

show()
#M.varschange(X3, S3)
#print M
#M.condition(S3, 0.8)
#M.varschange(S2, X3)
#M.varschange(X3, X1)
