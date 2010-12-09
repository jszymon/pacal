#os.system("E:\prog\python26_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf E:\m_\ecPro\pacal\demos\springer_book.py")
#os.system("D:\prog\python_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf D:\m_\ecPro\pacal\demos\demo.py")
#! Examples for distributions having different kinds of singularities 
#! ===================================================================
#!
#! The first import ``distr`` module.
#!
from functools import partial

from numpy import log
from pylab import figure, show

from pacal import *
from pacal.distr import demo_distr
#$
#$ Product of two normal variables is equal to $f_{N\cdotN}(z)=\pi^{-1}K_0(|z|)$
#$ where $K0$ is the modified Bessel function of the second kind
#$
figure()
demo_distr(NormalDistr(0,1) * NormalDistr(0,1))
#!
#! Product of two shifted normal variables
#!

figure()
demo_distr(NormalDistr(2,1) * NormalDistr(3,1))
figure()
demo_distr(NormalDistr(-2,1) / NormalDistr(3,2))
figure()
demo_distr(UniformDistr(-2,1) * NormalDistr(-1,3))
figure()
demo_distr(NormalDistr(1,1) / UniformDistr(-2,1))
figure()
demo_distr(NormalDistr(1,1) * NormalDistr(1,1))
figure()
demo_distr(NormalDistr(1,1) / NormalDistr(1,1))
figure()
demo_distr(NormalDistr(3,1) * NormalDistr(-5,1))
figure()
demo_distr(NormalDistr(-5,1) / NormalDistr(2,1))#), xmin = -50, xmax=50)
figure()
demo_distr(UniformDistr(1,3) * NormalDistr(-1,3))
figure()
demo_distr(NormalDistr(1,1) / UniformDistr(2,3))
figure()
demo_distr(NormalDistr(0.5,1) * NormalDistr(0.5,1))
figure()
demo_distr(UniformDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1) * NormalDistr(0,1)* NormalDistr(0,1) * NormalDistr(0,1))
def prod_uni_pdf(n, x):
    pdf = (-log(x)) ** (n-1)
    for i in xrange(2, n):
        pdf /= i
    return pdf
figure()
demo_distr(UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1), theoretical = partial(prod_uni_pdf, 6))
figure()
demo_distr(UniformDistr(0,1.1) * UniformDistr(0,1.1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1) * UniformDistr(0,1))

show()
