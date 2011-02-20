from pacal import *


fun = PiecewiseDistribution([])
fun.addSegment(Segment(0,1, lambda x:x))
fun.addSegment(Segment(1,2, lambda x:2-x))
distr = PDistr(fun)

# Here we have summary statistics and accuracies
distr.summary()
#! Another way to do it:
distr =  FunDistr(fun = lambda x: 1-abs(1-x), breakPoints=[0,1,2])
distr.summary()
#$ Notice that we set the break point at point 1 and User have to care 
#$ about $L_1$ nor of distribution.     

#! The same one obtain using sum of two uniform distributions
u = UniformDistr(0,1)
tri = u + u
tri.summary()

#! Users API for distr object:

from pylab import figure, show

#! Standard distributions
#! ----------------------
#$ Generally most kind of distributions one can obtain using above
#$ constructions, but we provide some standard distributions    
#$ \begin{description}
#$ \item[NormalDistr] normal distribution 
#$ \item[UniformDistr] uniform distribution
#$ \item[CauchyDistr] Cauchy distribution
#$ \item[ChiSquareDistr] $\chi^2$ distribution
#$ \item[ExponentialDistr] exponential distribution
#$ \item[GammaDistr] gamma distribution
#$ \item[BetaDistr] beta distribution
#$ \item[ParetoDistr] Pareto distribution
#$ \item[LevyDistr] Levy distribution
#$ \item[LaplaceDistr] Laplace distribution
#$ \item[StudentTDistr] Student's T distribution
#$ \item[SemicircleDistr] Semicircle distribution
#$ \item[FDistr] Snedecor's F distribution
#$ \item[FunDistr] user's defined distribution
#$ \item[DiscreteDistr] user's defined discrete distribution
#$ \end{description}

#! Standard distributions - examples
#! ---------------------------------
for d in [NormalDistr(), UniformDistr(), CauchyDistr(), ChiSquareDistr(),
          ExponentialDistr(), BetaDistr(), ParetoDistr(), LevyDistr(), LaplaceDistr(),
          StudentTDistr(), SemicircleDistr(), FDistr(), DiscreteDistr()]:
    figure()
    d.get_piecewise_cdf().plot(xmin = -3, xmax= 3, linewidth = 2.0, color = 'r')
    d.hist(xmin = -3, xmax= 3)
    d.plot(xmin = -3, xmax= 3, linewidth = 2.0, color = 'k')
    d.summary()
    show()
    
#! Arithmetics
#! -----------

#! Stable distributions
#! ````````````````````
N1 = NormalDistr()
N2 = NormalDistr()
Y1 = 0.5 * N1 + 0.5 * N2
Y2 = N1 / sqrt(2)
figure()
Y2.plot(xmin = -5, xmax =5, color = 'c', linewidth = 5)
Y1.plot(xmin = -5, xmax =5, color = 'k', linewidth = 1)
show()
N1.summary()
Y1.summary()
Y2.summary()
Y3 = 0.2 * N1 + 0.8 * N2
Y3.summary()

C1 = CauchyDistr()
C2 = CauchyDistr()
Y1 = 0.5 * C1 + 0.5 * C2
Y2 = 0.2 * C1 + 0.8 * C2
figure()
C1.plot(xmin = -10, xmax =10, color = 'c', linewidth = 7)
Y1.plot(xmin = -10, xmax =10, color = 'r', linewidth = 4)
Y2.plot(xmin = -10, xmax =10, color = 'k', linewidth = 1)
show()

L1 = LevyDistr()
L2 = LevyDistr()
Y1 = 0.5 * L1 + 0.5 * L2
Y2 = 2 * L1
figure()
L1.plot(xmin = 0, xmax =10, color = 'c', linewidth = 5)
Y1.plot(xmin = 0, xmax =10, color = 'r', linewidth = 3)
Y2.plot(xmin = 0, xmax =10, color = 'k', linewidth = 1)
show()
L1.summary()
Y1.summary()
Y2.summary()
Y3 = 0.2 * L1 + 0.8 * L2
Y3.summary()
#! Ratio distributions
#! ```````````````````
U = UniformDistr(0.5,2)
V = UniformDistr(0.5,2)
R = U / V
P = R * R
Q = R / R
figure()
P.plot(color = 'c', linewidth = 5)
Q.plot(color = 'k', linewidth = 1)
show()
P.summary()
Q.summary()
#! Discriptive statistics, random number generators, quantiles
#! -----------------------------------------------------------
fun = PiecewiseDistribution([])
fun.addSegment(Segment(0,1, lambda x: 0.5 + 0 * x))
fun.addSegment(Segment(1,2, lambda x: 2 - x))
X = PDistr(fun)
X.summary()
r = X.rand_invcdf(10)
print r
figure()
X.hist(n=10000, bins=10)
X.plot(linewidth = 2.0)
X.get_piecewise_cdf().plot(linewidth = 2.0)
show()
print "1-st quartile: ", X.quantile(0.25)
print "median: ", X.quantile(0.5)
print "3-rd quartile", X.quantile(0.75)

