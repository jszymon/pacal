#os.system("D:\prog\python_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf D:\m_\ecPro\pacal\demos\demo.py")
#! Discrete random variables using PaCal 
#! =========================================
#$ This demo ilustrates hot to use **PaCal** with discrite random variables.
#$
from pylab import figure, show, subplot

from pacal import *
from pacal.distr import demo_distr

#
#! Constuctor
#!------------
#!
U = UniformDistr(0,1)
M = DiscreteDistr(xi=[0.0, 1.0, 2.0, 3.0], pi=[0.3, 0.4, 0.2, 0.1])
S = M + U
SM = min(S, S)

#! 
M.summary()
S.summary()
SM.summary()

figure()
subplot(121)
S.plot()
subplot(122)
SM.plot()
figure()

M = DiscreteDistr(xi=[0.0, 1.0, 2.0, 3], pi=[0.42, 0.4, 0.13, 0.05])
print M.rand_invcdf(10, use_interpolated=True)
print M.rand_invcdf(10, use_interpolated=False)
M.summary()
N = M - 1
N.summary()
M.plot()
figure()
N.plot()

print "=========="
AM = M ** 2
AM.plot()
figure()
BM = abs(M)
BM.plot()
figure()
CM = M * 4
CM.plot()
CM.summary()
print CM.get_piecewise_pdf()
print CM.__class__

#! New plot
#!----------
figure()
S.get_piecewise_cdf().plot()
SM.get_piecewise_cdf().plot()

#! New plot
#!----------
figure()
I = OneDistr()
Two = I + I
Two.plot()
Two.get_piecewise_cdf().plot()
Two.hist()
I.hist()

#! Bernoulli distribution $B(k,5,0.8)$
#! -----------------------------------
b5 = ZeroDistr()
for i in range(5):
    b5 += DiscreteDistr(xi = [0, 1], pi = [0.2, 0.8])
figure()
subplot(131)
b5.plot()
subplot(132)
b5.get_piecewise_cdf().plot()
b5.summary()
subplot(133)
b5.hist()

d = DiscreteDistr(xi = [1, 2], pi = [0.2, 0.8])
U = UniformDistr(0,2)
A1 = d + U
A2 = d * U
A3 = d / U
A4 = U / d
figure()
subplot(221)
A1.plot()
subplot(222)
A2.plot()
subplot(223)
A3.plot(xmax=6.0)
subplot(224)
A4.plot()
subplot(221)
A1.hist()
subplot(222)
A2.hist()
subplot(223)
A3.hist(xmin=0.5, xmax=6)
subplot(224)
A4.hist()

A1.summary()
A2.summary()
A3.summary()
A4.summary()


figure()
A1.get_piecewise_cdf().plot()
A2.get_piecewise_cdf().plot()
A3.get_piecewise_cdf().plot(xmax=6.0)
A4.get_piecewise_cdf().plot()

show()
