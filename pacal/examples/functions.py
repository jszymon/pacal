#os.system("E:\prog\python26_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf E:\m_\ecPro\pacal\pacal\\examples\\springer_book.py")
#os.system("E:\prog\python26_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf E:\m_\ecPro\pacal\pacal\\examples\\functions.py")
#os.system("E:\prog\python26_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf E:\m_\ecPro\pacal\pacal\\examples\\singularities.py")
#os.system("D:\prog\python_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf D:\m_\ecPro\pacal\demos\demo.py")
#! Functions of random variables using PaCal 
#! =========================================
#!
#! This demo shows how to use **PaCal** functions of random variables.
#!
#! The first import ``distr`` module.
#!
from pylab import figure, show
from numpy import pi

from pacal import *

from pacal.distr import demo_distr

#!
#! Inverse of r.v. 
#! ---------------

U = UniformDistr(0.0,1.0)
N = NormalDistr(0,1)

d = 1/(N**2)**0.5
#d.summary()
C = CauchyDistr()
(2/C).summary()
(C/2).summary()
demo_distr(C/2, 0.5/C)

E1 = ChiSquareDistr(1)
E2 = ChiSquareDistr(3)
figure()
demo_distr(log(E1))
figure()
demo_distr(log(E2))
figure()
demo_distr(log(UniformDistr(0.0,1.0)))
figure()
demo_distr(exp(UniformDistr(0.0,1.0)))
figure()
demo_distr(exp(UniformDistr(0.0,1.0))+exp(UniformDistr(0.0,1.0)))
figure()
demo_distr(log(UniformDistr(0.0,1.0))+log(UniformDistr(0.0,1.0)))
figure()
demo_distr(log(E1)+log(E2))

try:
    l = log(NormalDistr(0, 1))
    print "Error: log of negative distribution not reported!!!!"
except ValueError:
    pass

#! Following operations are permitted:
O = ZeroDistr()
I = OneDistr()
#Two = I+I
#MinusOne = -I
#A = abs(C)
#A = abs(C)
#S = C**2
invC = I/C
figure()
demo_distr(invC, C)

figure()
N2=min(I, N)
 

figure()
demo_distr(abs(C + 2), xmax = 10)
figure()
demo_distr((C+1)**2, xmax = 1, ymax=10)
#! This figure is correct!
figure()
demo_distr(1/C, theoretical = C, xmin =-10, xmax = 10)


figure()
demo_distr(sqrt(E1), theoretical = abs(NormalDistr()), xmin =0, xmax = 10,  ymax=1.1)
figure()
demo_distr(sqrt(E2), xmin =0, xmax = 10)

#!

#!
#! Arcus tangent of Cauchy distribution 
#! -------------------------------------

C = NormalDistr(0,1) / NormalDistr(0,1)
U = atan(C); 
figure()
demo_distr(U, theoretical = UniformDistr(-pi/2, pi/2), histogram = True)


C_shifted = NormalDistr(1.2,1) / NormalDistr(1.2,1)
figure()
demo_distr(C_shifted, theoretical = None, histogram = True, xmin=-5, xmax=5)
#show()

U_shifted = atan(C_shifted); 
figure()
demo_distr(U_shifted, theoretical = None, histogram = True)
#show()

#L.summary()
#figure()
#L.plot()
#histdistr(L)
show()
