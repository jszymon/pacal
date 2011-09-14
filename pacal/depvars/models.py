'''
Created on 07-12-2010

@author: marcin
'''
import sys
import time
import sympy
import traceback
from numpy import size, isnan
from pylab import *
import pacal

from pacal.distr import *
from pacal.standard_distr import *
from pacal.segments import PiecewiseDistribution#, _segint
from pacal.depvars.copulas import GumbelCopula, GumbelCopula2d, ClaytonCopula, FrankCopula2d, FrankCopula
from pacal.depvars.copulas import PiCopula, WCopula, MCopula

from pacal.depvars.nddistr import NDFun
from pacal.depvars.nddistr import plot_2d_distr

class Model(object):
    def __init__(self, nddistr, rvs=[]):
        self.nddistr = nddistr
        self._segint = nddistr._segint 
        free_rvs = set(self.nddistr.Vars)
        # fetch all necessary RVs
        dep_rvs = set(rvs)
        for rv in rvs:
            dep_rvs.update(rv.getParentsDep())
        self.free_rvs = free_rvs
        self.dep_rvs = dep_rvs
        self.all_vars = free_rvs | dep_rvs
        self.sym_to_rv = {}
        for rv in self.all_vars:
            self.sym_to_rv[rv.getSymname()] = rv
        self.rv_to_equation = {}
        for rv in self.dep_rvs:
            self.rv_to_equation[rv] = rv.getSym()
    def __str__(self):
        s = "Model:\n"
        s += "free vars: " + ", ".join(str(rv.getSymname())+ "(" + str(self.eval_var(rv)) +")" for rv in self.free_rvs) + "\n"
        s += "dep vars:  " + ", ".join(str(rv.getSymname())+ "(" + str(self.eval_var(rv)) +")" for rv in self.dep_rvs) + "\n"
        s += "Equations:\n"
        for rv, eq in self.rv_to_equation.iteritems():
            s += str(rv.getSymname()) + " = " + str(eq) + "(" + str(self.eval_var(rv)) + ")\n"
        s += "\n"
        self.toGraphwiz()
        return s

    def prepare_var(self, var):
        if isinstance(var, basestring):
            var = self.sym_to_rv[sympy.Symbol(var)]
        elif isinstance(var, sympy.Symbol):
            var = self.sym_to_rv[var]
        return var
    def varschange(self, free_var, dep_var):
        free_var = self.prepare_var(free_var)
        dep_var = self.prepare_var(dep_var)
        if not self.is_free(free_var):
            raise RuntimeError("First exchanged variable must be free")
        if not self.is_dependent(dep_var):
            raise RuntimeError("First exchanged variable must be free")
        parents = self.rv_to_equation[dep_var].atoms(sympy.Symbol)
        if free_var.getSymname() not in parents:
            raise RuntimeError("Free variable is not a parent of dependent variable")
        for s in parents:
            if not self.is_free(s):
                raise RuntimeError("Dependent variable has a nonfree parent")
        inv_transf, inv_transf_lamdified, inv_transf_vars, jacobian = self.var_change_helper(free_var, dep_var)
        self.free_rvs.remove(free_var)
        self.free_rvs.add(dep_var)
        self.dep_rvs.add(free_var)
        self.dep_rvs.remove(dep_var)
        del self.rv_to_equation[dep_var]
        for rv, eq in self.rv_to_equation.iteritems():
            if free_var.getSymname() in set(eq.atoms(sympy.Symbol)):
                self.rv_to_equation[rv] = eq.subs(free_var.getSymname(), inv_transf)
        self.rv_to_equation[free_var] = inv_transf
        self.nddistr = self.nddistr.varschange(free_var, inv_transf_lamdified, inv_transf_vars, jacobian)

    def var_change_helper(self, free_var, dep_var):
        """Compute inverse transformation and Jacobian for substituting
        dep_var for free_var."""
        free_var = self.prepare_var(free_var)
        dep_var = self.prepare_var(dep_var)
        # inverve transformation
        uj = sympy.solve(self.rv_to_equation[dep_var] - dep_var.getSymname(), free_var.getSymname())
        assert len(uj) == 1, uj
        uj = uj[0]
        uj_symbols = list(sorted(uj.atoms(sympy.Symbol)))
        inv_transf = sympy.lambdify(uj_symbols, uj, "numpy")  
        inv_transf_vars = [self.sym_to_rv[s] for s in uj_symbols]

        print "vars to change: ", free_var.getSymname(), " <- ", dep_var.getSymname(), "=", self.rv_to_equation[dep_var]
        print "equation: ", dep_var.getSymname(), "=", self.rv_to_equation[dep_var]
        print "solution: ", free_var.getSymname(), "=", uj
        print "variables: ", uj_symbols, inv_transf_vars

        # Jacobian
        #J = sympy.Abs(sympy.diff(uj, dep_var.getSymname()))
        J = sympy.diff(uj, dep_var.getSymname())
        J_symbols = list(sorted(J.atoms(sympy.Symbol)))
        if len(J_symbols) > 0:
            jacobian_vars = [self.sym_to_rv[s] for s in J_symbols]
            jacobian = sympy.lambdify(J_symbols, J, "numpy")
            jacobian = NDFun(len(jacobian_vars), jacobian_vars, jacobian, safe = True, abs = True)
        else:
            jacobian = NDConstFactor(abs(float(J)))
            jacobian_vars = []

        print "J=", J
        print "variables: ", J_symbols, jacobian_vars

        return uj, inv_transf, inv_transf_vars, jacobian
        

    def eliminate(self, var):
        var = self.prepare_var(var)
        if var in self.free_rvs:
            for rv, eq in self.rv_to_equation.iteritems():
                if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                    raise RuntimeError("Cannot eliminate free variable on which other variables depend")
            self.nddistr = self.nddistr.eliminate(var)
            self.free_rvs.remove(var)
        elif var in self.dep_rvs:
            subs_eq = self.rv_to_equation[var]
            del self.rv_to_equation[var]
            self.dep_rvs.remove(var)
            for rv, eq in self.rv_to_equation.iteritems():
                if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                    self.rv_to_equation[rv] = eq.subs(var.getSymname(), subs_eq)
        else:
            assert False
    def eliminate_other(self, vars):
        vars_to_eliminate = self.dep_rvs - set(vars)
        for var in vars_to_eliminate:
            self.eliminate(var)
    def varchange_and_eliminate(self, var):
        if self.is_free(var) and len(self.dep_rvs)==0:
            # eliminujemy pozostale zmienne wolne
            pass
        elif self.is_free(var) and len(self.dep_rvs)==0:
            # eliminujemy zbedne zmienne zalezne od konca
            pass
        elif not self.is_free(var):
            # wybieramy najlepsza znienna zalezna i podmieniamy z wolna
            pass
    def eval_var(self, var):        
        var = self.prepare_var(var)
        note = 0
        if self.is_dependent(var):
            eq = self.rv_to_equation[var]
            for a in eq.atoms(sympy.Symbol):
                if self.is_free(a):
                    note += 1
                elif self.is_dependent(a):
                    note += 10
                else: 
                    note += 1000
        else:
            for rv, eq in self.rv_to_equation.iteritems():
                if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                    note += 1
        return note                    
               
    def condition(self, var, X, **kwargs):
        var = self.prepare_var(var)
        if not self.is_free(var):
            raise RuntimeError("You can only condition on free variables")
        Xsym = sympy.S(X)
        for rv, eq in self.rv_to_equation.iteritems():
            if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                self.rv_to_equation[rv] = eq.subs(var.getSymname(), Xsym)
        self.free_rvs.remove(var)
        self.nddistr = self.nddistr.condition([var], X)

    def as1DDistr(self):
        if len(self.dep_rvs) > 0:
            raise RuntimeError("Cannot get distribution of dependent variable.")
        if len(self.free_rvs) != 1:
            raise RuntimeError("Too many free variables")
        pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
        return pfun
    
    def summary(self):
        if len(self.free_rvs) == 1:
            pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
            pfun.summary()
        else:
            raise RuntimeError("Too many variables.")        
    
    def plot(self, **kwargs):
        if len(self.free_rvs) == 1:
            pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
            pfun.plot(label = str(self.nddistr.Vars[0].getSymname()), **kwargs)
            legend()
            pfun.summary()
        elif len(self.free_rvs) == 2:
            plot_2d_distr(self.nddistr)
        else:
            raise RuntimeError("Too many variables.")
    def is_free(self, var):
        var = self.prepare_var(var)
        return var in self.free_rvs
    
    def is_dependent(self, var):
        var = self.prepare_var(var)
        return var in self.dep_rvs
    
    def toGraphwiz(self, f = sys.stdout):
        print >>f, "graph G {rankdir = BT"
        for key in self.free_rvs:
            print >>f, "\"{0}\"".format(key.getSymname()), " [label=\"{0}\"]".format(key.getSymname())
        for key in self.dep_rvs:
            print >>f, "\"{0}\"".format(key.getSymname()), " [label=\"{0}={1}\"]".format(key.getSymname(),self.rv_to_equation[key])
        for rv, eq in self.rv_to_equation.iteritems():
            for a in eq.atoms(sympy.Symbol):
                print >>f, "\"{0}\" -- \"{1}\"".format(str(rv.getSymname()), str(a))
        print >>f, "}"
            
class TwoVarsModel(Model):    
    """Two dimensional model with one equation""" 
    def __init__(self, nddistr=None, d=None):
        super(TwoVarsModel, self).__init__(nddistr, [d])
        self.eliminate_other([d])
        self.d = d
        self.vars = []
        self.symvars = []
        for var in nddistr.Vars: #self.free_rvs:
            self.vars.append(var)
            self.symvars.append(var.getSymname()) 
        print "=====", self.vars
        print self.symvars
        print self.dep_rvs
        print self.rv_to_equation
        self.symop = self.rv_to_equation[d]
        
        if self.vars.__len__() <> 2:
            raise Exception("use it with two variables")
        x = self.symvars[0]
        y = self.symvars[1]
        z = sympy.Symbol("z")
        self.fun_alongx = sympy.solve(self.symop - z, y)[0]
        self.fun_alongy = sympy.solve(self.symop - z, x)[0]
        
        self.lfun_alongx = sympy.lambdify([x, z], self.fun_alongx)    
        self.lfun_alongy = sympy.lambdify([y, z], self.fun_alongy)
        self.Jx = 1 * sympy.diff(self.fun_alongx, z)
        print "Jx=", self.Jx
        print "fun_alongx=",self.fun_alongx
        self.Jy = 1 * sympy.diff(self.fun_alongy, z)
        self.lJx = sympy.lambdify([x, z], self.Jx)
        self.lJy = sympy.lambdify([y, z], self.Jy)
        self.z = z
    def solveCutsX(self, fun, ay, by):        
        axc = sympy.solve(fun - ay, self.symvars[0])[0]
        bxc = sympy.solve(fun - by, self.symvars[0])[0]
        return (axc, bxc)
    def solveCutsY(self, fun, ax, bx):        
        #ayc = sympy.solve(fun - ay, self.z)[0]
        #byc = sympy.solve(fun - by, self.z)[0]
        ayc = fun.subs(self.symvars[0], ax)
        byc = fun.subs(self.symvars[0], bx)
        return (ayc, byc)
    def getUL(self, ax, bx, ay, by, z):
        axcz, bxcz = self.solveCutsX(self.fun_alongx , ay, by)
        axc = axcz.subs(self.z, z)
        bxc = bxcz.subs(self.z, z)
        if not axc.is_real:
            axc = None
        if not axc.is_real:
            bxc = None
        aycz, bycz = self.solveCutsY(self.fun_alongx, ax, bx)
        ayc = aycz.subs(self.z, z)
        byc = bycz.subs(self.z, z)
        if not axc.is_real:
            byc = None
        if not axc.is_real:
            byc = None
        #print "========"
        #print ay, ayc, by
        #print ay, byc, by
        #print ax,bx,axc, bxc 
        if (ay < ayc and ayc < by):
            L = ax
        else:
            L = min(max(axc, ax), min(bxc, bx))
        if (ay < byc and byc < by):
            U = bx
        else:
            U = max(max(axc, ax), min(bxc, bx))
        if L < U:
            pass 
        else:
            L, U = U, L 
        return max(ax, L), min(bx, U) 
    
    def plotFrame(self, ax, bx, ay, by):
        plt.figure()
        h = 0.1
        plt.axis((ax - h, bx + h, ay - h, by + h))
        print self.symvars
         
        print self.d.getSym()
        print self.symop
        x = self.symvars[0]
        y = self.symvars[1]
        
        lop = sympy.lambdify([x, y], self.symop) 
        tmp = [lop(ax, ay), lop(ax, by), lop(bx, ay), lop(bx, by)]
        print tmp
        i0, i1 = min(tmp), max(tmp)
        for i in linspace(i0, i1, 20):
            #y =  self.lfun_alongx(t, i)
            #plot(t, y)
            try:
                L, U = self.getUL(ax, bx, ay, by, i)
                #print "**", i, L, U
                tt = linspace(L, U, 100)
                #print self.fun_alongx
                y = self.lfun_alongx(tt, array([i]))
                plot(tt, y, "k", linewidth=2.0)
                
                #axcz, bxcz = self.solveCutsX(self.fun_alongx , ay,by)
                ##print "==", self.fun_alongx, "|", axcz, "|", bxcz
                #axc = axcz.subs(self.z, i)
                #bxc = bxcz.subs(self.z, i)
                ##print "==", axc,bxc
                #plot(axc,ay, 'k.')
                #plot(bxc,by, 'b.')
                
                #aycz, bycz = self.solveCutsY(self.fun_alongx, ax,bx)
                
                ##print "====", axcz,bxcz 
                #ayc = aycz.subs(self.z, i)
                #byc = bycz.subs(self.z, i)
                ##print "====", ayc,byc
                #plot(ax, ayc, 'r.')
                #plot(bx, byc, 'g.')
            
                
            except:
                traceback.print_exc()  
        plot([ax, ax], [ay, by], "k:")
        plot([bx, bx], [ay, by], "k:")
        plot([ax, bx], [ay, ay], "k:")
        plot([ax, bx], [by, by], "k:")
        plt.plot()
        pass
        
        
    #def getAlongX(self):
    #        z = var('z');
    #        print "symop=", d.get
    
    def convmodel(self):
        """Probabilistic operation defined by model
        """
        op = self.symop#d.getSym()
        x = self.symvars[0]
        y = self.symvars[1]
        lop = sympy.lambdify([x, y], op) 
        F = self.vars[0]
        G = self.vars[1]
        #self.nddistr.setMarginals(F, G)
        f = self.vars[0].get_piecewise_pdf()
        g = self.vars[1].get_piecewise_pdf()
        bf = f.getBreaks()
        bg = g.getBreaks()
        
        bi = zeros(len(bf) * len(bg))
        k = 0;
        for xi in bf:
            for yi in bg:
                if not isnan(lop(xi, yi)):
                    bi[k] = lop(xi, yi)
                else:
                    print "not a number, xi=", xi, "yi=", yi, "result=", lop(xi,yi)
                k += 1
        ub = array(unique(bi))
        
        fun = lambda x : self.convmodelx(segList, x)            
        fg = PiecewiseDistribution([]);
        
        if isinf(ub[0]):
            segList = _findSegList(f, g, ub[1] -1, lop)
            seg = MInfSegment(ub[1], fun)
            segint = seg.toInterpolatedSegment()
            fg.addSegment(segint)
            ub=ub[1:]
        if isinf(ub[-1]):
            segList = _findSegList(f, g, ub[-2] + 1, lop)
            seg = PInfSegment(ub[-2], fun)
            segint = seg.toInterpolatedSegment()
            fg.addSegment(segint)
            ub=ub[0:-1]
        print "f=", f
        print "g=", g
        print "=======", ub
        for i in range(len(ub) - 1) :
            segList = _findSegList(f, g, (ub[i] + ub[i + 1]) / 2, lop)
            seg = Segment(ub[i], ub[i + 1], partial(self.convmodelx, segList))
            #seg = Segment(ub[i],ub[i+1], fun)
            segint = seg.toInterpolatedSegment()
            fg.addSegment(segint)
    
        # Discrete parts of distributions
        #fg_discr = convdiracs(f, g, fun = lambda x,y : x * p + y * q)
        #for seg in fg_discr.getDiracs():
        #    fg.addSegment(seg)
        return fg

    def convmodelx(self, segList, xx):
        """Probabilistic weighted mean of f and g, integral at points xx 
        """    
        op = self.symop#d.getSym()
        x = self.symvars[0]
        y = self.symvars[1]
        lop = sympy.lambdify([x, y], op) 
        
        if size(xx) == 1:
            xx = asfarray([xx])
        wyn = zeros_like(xx)
        
        P = self.nddistr
        #P.setMarginals(F,G)
        #fun = lambda t : P.cdf(t, self.lfun_alongx(t, array([zj]))) * abs(self.lJx(t, array([zj])))
        if isinstance(P, pacal.depvars.copulas.MCopula) | isinstance(P, pacal.depvars.copulas.WCopula):
            #print ">>", P
            #funPdf = lambda t : P.cdf(t, self.lfun_alongx(t, zj)) * abs(self.lJx(t, zj))
            funCdf = lambda t : P.cdf(t, self.lfun_alongx(t, zj)) * abs(self.lJx(t, zj))
            fun = funCdf
        elif isinstance(P, pacal.depvars.copulas.PiCopula):        
            fun = lambda t : segi(t) * segj(self.lfun_alongx(t, array([zj]))) * abs(self.lJx(t, array([zj])))             
            #print "here"
        else:
            #print "<<<", P  
            fun = lambda t : P.pdf(t, self.lfun_alongx(t, zj)) * abs(self.lJx(t, zj))
        ##fun = lambda t : P.jpdf_(F, G, t, self.lfun_alongx(t, array([zj])))  * abs(self.lJx(t, array([zj])))

        
        for j in range(len(xx)) :  
            zj = xx[j]
            if isinstance(P, pacal.depvars.copulas.WCopula):
                I = 1
            else:
                I = 0
            err = 0
            #print j
            for segi, segj in segList:
                if segi.isSegment() and segj.isSegment():
                    L, U = self.getUL(segi.a, segi.b, segj.a, segj.b, zj)
                    L, U  = sort([U, L])
                    X = NormalDistr(0,1, sym="X")
    #tt = linspace(L, U, 100) 
                    ##print self.fun_alongx
                    #y =  self.lfun_alongx(tt,zj)
                    #plot(tt, y, "k", linewidth=2.0)
                
                    if L < U:
                        #print zj, L, U#, fun
                        i, e = self._segint(fun, float(L), float(U))   
                        print j, zj, i                     
                    else:  
                        i, e = 0, 0
                #elif segi.isDirac() and segj.isSegment():
                #    i = segi.f*segj((x-segi.a)/q)/q   # TODO 
                #    e=0;
                #elif segi.isSegment() and segj.isDirac():
                #    i = segj.f*segi((x-segj.a)/p)/p   # TODO
                #    e=0;
                #elif segi.isDirac() and segj.isDirac():
                #    pass
                    #i = segi(x-segj.a)/p/q          # TODO
                    #e=0;
                if isinstance(P, pacal.depvars.copulas.WCopula):
                    I = min(I,i)
                elif isinstance(P, pacal.depvars.copulas.MCopula):
                    I = max(I,i)
                else:
                    I += i
                err += e
            wyn[j] = I
        return wyn
    
    def varchange_and_eliminate(self):
        return PDistr(self.convmodel())
            
def _findSegList(f, g, z, op):
    """It find list of segments for integration purposes, for given z 
    input: f, g - piecewise function, z = op(x,y), op - operation (+ - * /)
    output: list of segment products depends on z 
    """
    slist = [];
    for segi in f.segments:
        for segj in g.segments: 
            R1 = array([segi.a, segi.b, segi.a, segi.b]) 
            R2 = array([segj.a, segj.b, segj.b, segj.a]) 
            R = op(R1, R2)
            R = unique(R[isnan(R) == False])
            if min(R) < z < max(R):
                slist.append((segi, segj))    
    return slist
    #fun = lambda t : segi( t ) * segj( lxfun(t,xj) )# * abs(lJy(t,xj))



if __name__ == "__main__":
    from pacal.distr import demo_distr
    from pacal.depvars.nddistr import *
    #X = UniformDistr(1, 2, sym="x1")
    #Y = UniformDistr(1, 3, sym="x2")
    
    X1 = UniformDistr(1.5, 2.5, sym="x1")
    X2 = UniformDistr(1.5, 2.5, sym="x2")
    X3 = UniformDistr(1.5, 2.5, sym="x3")
    X4 = UniformDistr(1.5, 2.5, sym="x4")
    X5 = UniformDistr(1.5, 2.5, sym="x5")
    X6 = UniformDistr(1.5, 2.5, sym="x6")

    #Y = UniformDistr(1.5, 2.5, sym=sympy.Symbol("Y"))
    #X = UniformDistr(0.5, 1.5, sym=sympy.Symbol("X"))
    #Z = BetaDistr(1.5,1.5, sym = sympy.Symbol("Z"))
    X = NormalDistr(0,1, sym="X")
    Y = NormalDistr(2, 3, sym="Y")
    Z = BetaDistr(2, 3, sym="Z")
    
    X = NormalDistr(sym="X")
    Y = ExponentialDistr(sym="Y")
    
    #X = UniformDistr(0, 1, sym="x1")
    #Y = UniformDistr(0, 2, sym="x2")
    
#    # ==== probability boxex ===============================
#    cw = WCopula(marginals=[X, Y])
#    cw.plot()
#    cm = MCopula(marginals=[X, Y])
#    cm.plot()
#    #show()
#    cp = PiCopula(marginals=[X, Y])
#    U = X + Y #/ (Y + 1)# * X
#    
#    Mw = TwoVarsModel(cw, U)
#    Mm = TwoVarsModel(cm ,U)
#    #Mp = TwoVarsModel(cp ,U)
#    
#    funw = Mw.varchange_and_eliminate()
#    funm = Mm.varchange_and_eliminate()
#    #funp = Mp.varchange_and_eliminate()
#    figure()
#    funw.plot()
#    funw.summary()
#    funm.plot()
#    funp.get_piecewise_cdf().plot()
#    funp.summary()
    
    
    
#    for theta in [5, 10]:
#        print "::", theta
#        ci = GumbelCopula2d(marginals=[X, Y], theta=theta)
#        Mi = TwoVarsModel(ci, U)
#        funi = Mi.varchange_and_eliminate()
#        funi.get_piecewise_cdf().plot(color="g")
#        funi.summary()
#    for theta in [-15, -5, 5, 15]:
#        print "::::", theta
#        ci = FrankCopula2d(marginals=[X, Y], theta=theta)
#        Mi = TwoVarsModel(ci, U)
#        funi = Mi.varchange_and_eliminate()
#        funi.get_piecewise_cdf().plot(color="b")
#        funi.summary()
#    for theta in [5, 10]:
#        print ":::", theta
#        ci = ClaytonCopula(marginals=[X, Y], theta=theta)
#        Mi = TwoVarsModel(ci, U)
#        funi = Mi.varchange_and_eliminate()
#        funi.get_piecewise_cdf().plot(color="r")
#        funi.summary()
#    print "==============="
#    V= X-Y
#    cp = PiCopula(marginals=[X, Y])
#    m = TwoVarsModel(cp, V)
#    fun = m.varchange_and_eliminate()
#    fun.summary()
#    fun.plot()
#    show()
#    0/0
    
    
    cij = IJthOrderStatsNDDistr(X, 8, 2, 7)
    X1, X2 = cij.Vars
    plot_2d_distr(cij)
    figure()

    X1.plot(color="r")
    X1.summary()
    X2.plot(color="g")
    X2.summary()
    
    V=X2-X1
    
    print "p=", V.parents[1].getSym()
    mR = TwoVarsModel(cij, V)
    funR = mR.varchange_and_eliminate()
    funR.summary()
    funR.plot(color="k")
    
#    cc = ClaytonCopula(marginals=[X1, X2], theta=1.0/10.0)
#    cc.plot()
#    mC = TwoVarsModel(cc,V)
#    funC = mC.varchange_and_eliminate()
#    funC.summary()
#    funC.plot(color="m")
    
    K = V
    K.plot(color="b")
    K.summary()
    
    show()
    0/0
        
    
    #funm.summary()
    show()
    0/0 
    
    # ====================================================

