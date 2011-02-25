#os.system("D:\prog\python_packages\pyreport-0.3.4c\pyreport\pyreport.py -e -l -t pdf D:\m_\ecPro\pacal\demos\demo.py")
#! Discrete random variables using PaCal 
#! =========================================
#$ This demo illustrates hot to use **PaCal** with discrite random variables.
from pacal import *
from pylab import *

#! New plot
#!-------------------
#$ To jest \LaTeX $e^{i\pi} -1=0$
#!* 1s D4 dfs
#!** 3ddsf  f
#!* 4 D4 s ds
#!
#$ This is \LaTeX : $c = 2\cdot(a+b)$
#!
import time
t0 =time.time()
U = LevyDistr()
S = U
for i in range(20):
    S = S+U
S.summary()
print "done ", time.time()-t0 
#! Bernoulli distribution $B(k,5,0.8)$
#! -----------------------------------
figure()
D = DiscreteDistr(xi =[0, 1], pi = [0.4, 0.6])
D.plot()
D.summary()

b5 =  ZeroDistr()
for i in range(15):
    b5 +=  D
figure()
subplot(121)
b5.plot()
subplot(122)
b5.get_piecewise_cdf().plot()
b5.summary()

#! Discrete and continuous variables
#!----------------------------------
 
D4 = DiscreteDistr(xi =[1, 2, 3, 4], pi = [0.25, 0.5, 0.15, 0.1])
U = UniformDistr(0,2)
figure()
subplot(121)
D4.plot()
title(D4.getName())
subplot(122)
U.plot()
title(U.getName())

A1 = D4 + U
A2 = D4 * U
A3 = D4 / U
A4 = U / D4
figure()
subplot(221)
A1.plot()
A1.hist(bins = 200)
title(A1.getName())
subplot(222)
A2.plot()
A2.hist(bins = 200)
title(A2.getName())
subplot(223)
A3.plot(xmax=4.0)
A3.hist(xmax=4.0, bins = 200)
title(A3.getName())
subplot(224)
A4.plot()
A4.hist(bins = 200)
title(A4.getName())

A1.summary()
A2.summary()
A3.summary()
A4.summary()


figure()
subplot(221)
A1.get_piecewise_cdf().plot()
title(A1.getName()+ " cdf")
subplot(222)
A2.get_piecewise_cdf().plot()
title(A2.getName()+ " cdf")
subplot(223)
A3.get_piecewise_cdf().plot(xmax=6.0)
title(A3.getName()+ " cdf")
subplot(224)
title(A4.getName()+ " cdf")
A4.get_piecewise_cdf().plot()


#! Min and Max operations
#! ----------------------
U = UniformDistr(0,1)
M = DiscreteDistr(xi=[0.0, 1.0, 2.0, 3.0], pi=[0.3, 0.4, 0.2, 0.1])
S = M + U
SM = min(S, S)
#! 
S.summary()
SM.summary()

figure()
subplot(121)
S.plot()
subplot(122)
SM.plot()

figure()
S.get_piecewise_cdf().plot()
SM.get_piecewise_cdf().plot()

show()
