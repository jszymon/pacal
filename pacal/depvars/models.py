'''
Created on 07-12-2010

@author: marcin
'''

from __future__ import print_function

import sys
from copy import copy
from functools import partial
import traceback

from numpy import size, isnan, linspace, zeros, array, unique, isinf, zeros_like
from numpy import asfarray
import sympy
from pylab import legend, figure, plot, axis

from pacal.sympy_utils import eq_solve
from pacal.sympy_utils import my_lambdify
from pacal.standard_distr import FunDistr, PDistr
from pacal.segments import PiecewiseDistribution, PInfSegment, MInfSegment, DiracSegment, Segment
from pacal.depvars.nddistr import NDFun, NDConstFactor, NDProductDistr
from pacal.depvars.nddistr import plot_2d_distr, plot_1d1d_distr
from pacal import params

try:
    import pygraphviz as pgv
    have_pgv = True
except:
    have_pgv = False


def _get_str_sym_name(v):
    return str(v.getSymname())
class Model(object):
    def __init__(self, nddistr, rvs=[]):
        if not isinstance(nddistr, NDFun):
            nddistr = NDProductDistr(nddistr)
        self.nddistr = nddistr
        self._segint = nddistr._segint
        free_rvs = list(self.nddistr.Vars)
        # fetch all necessary RVs
        dep_rvs = set(rvs)
        for rv in rvs:
            dep_rvs.update(rv.getParentsDep())
        self.free_rvs = free_rvs
        self.dep_rvs = list(sorted(dep_rvs, key = _get_str_sym_name))
        self.all_vars = list(sorted(set(free_rvs) | dep_rvs, key = _get_str_sym_name))
        self.sym_to_rv = {}
        for rv in self.all_vars:
            self.sym_to_rv[rv.getSymname()] = rv
        self.rv_to_equation = {}
        for rv in self.dep_rvs:
            self.rv_to_equation[rv] = rv.getSym()
    def copy(self):
        cp = copy(self)
        cp.free_rvs = copy(cp.free_rvs)
        cp.dep_rvs = copy(cp.dep_rvs)
        cp.all_vars = copy(cp.all_vars)
        cp.sym_to_rv = copy(cp.sym_to_rv)
        cp.rv_to_equation = copy(cp.rv_to_equation)
        return cp
    def __str__(self):
        s = "Model:\n"
        #s += "free vars: " + ", ".join(str(rv.getSymname())+ "(" + str(self.eval_var(rv)) +")" for rv in self.free_rvs) + "\n"
        s += "free_vars:\n"
        for rv in self.free_rvs:
            s += "   " + str(rv.getSymname()) + " ~ " + str(rv.getName()) + "\n"
        s += "dep vars:  " + ", ".join(str(rv.getSymname()) for rv in self.dep_rvs) + "\n"
        s += "Equations:\n"
        for rv, eq in self.rv_to_equation.items():
            s += str(rv.getSymname()) + " = " + str(eq) + "(" + str(self.eval_var(rv)) + ")\n"
        s += str(self.nddistr)
        s += "\n"
        #self.toGraphwiz()
        return s

    def prepare_var(self, var):
        if isinstance(var, str):
            var = self.sym_to_rv[sympy.Symbol(var)]
        elif isinstance(var, sympy.Symbol):
            var = self.sym_to_rv[var]
        return var
    def get_children(self, var):
        """Children of a variable."""
        ch = []
        vsym = var.getSymname()
        for rv, eq in self.rv_to_equation.items():
            if vsym in set(eq.atoms(sympy.Symbol)):
                ch.append(rv)
        return ch
    def get_parents(self, var):
        pnames = self.rv_to_equation[var].atoms(sympy.Symbol)
        return [self.sym_to_rv[pn] for pn in pnames]

    def varschange(self, free_var, dep_var):
        free_var = self.prepare_var(free_var)
        dep_var = self.prepare_var(dep_var)
        if params.models.debug_info:
            print("exchange free variable: ", free_var.getSymname(), "with dependent variable", dep_var.getSymname())
        if not self.is_free(free_var):
            raise RuntimeError("First exchanged variable must be free")
        if not self.is_dependent(dep_var):
            raise RuntimeError("First exchanged variable must be free, second must be dependent")
        parents = self.rv_to_equation[dep_var].atoms(sympy.Symbol)
        if free_var.getSymname() not in parents:
            raise RuntimeError("Free variable is not a parent of dependent variable")
        for s in parents:
            if not self.is_free(s):
                raise RuntimeError("Dependent variable has a nonfree parent")
        var_changes, equation = self.var_change_helper(free_var, dep_var)
        if len(var_changes) != 1:
            print("Equation:", equation, "has multiple solutions")
            for vc in var_changes:
                print(vc[0])
            raise RuntimeError("Equations with multiple solutions are not supported")
            #var_changes = var_changes[:1]

        inv_transf, inv_transf_lamdified, inv_transf_vars, jacobian = var_changes[0]
        self.free_rvs.remove(free_var)
        self.free_rvs.append(dep_var)
        self.dep_rvs.append(free_var)
        self.dep_rvs.remove(dep_var)
        del self.rv_to_equation[dep_var]
        for rv, eq in self.rv_to_equation.items():
            if free_var.getSymname() in set(eq.atoms(sympy.Symbol)):
                self.rv_to_equation[rv] = eq.subs(free_var.getSymname(), inv_transf)
        self.rv_to_equation[free_var] = inv_transf
        #print str(free_var.getSymname())+ "=" + str(var_changes[0][0])
        self.nddistr = self.nddistr.varschange(free_var, inv_transf_lamdified, inv_transf_vars, jacobian)

    def var_change_helper(self, free_var, dep_var):
        """Compute inverse transformation and Jacobian for substituting
        dep_var for free_var."""
        free_var = self.prepare_var(free_var)
        dep_var = self.prepare_var(dep_var)
        # inverve transformation
        equation = self.rv_to_equation[dep_var] - dep_var.getSymname()
        solutions = eq_solve(self.rv_to_equation[dep_var], dep_var.getSymname(), free_var.getSymname())
        var_changes = []
        for uj in solutions:
            # remove complex valued solutions
            vj = uj.atoms(sympy.Symbol)
            hvj = {}
            for v in vj:
                #print self.sym_to_rv[v].range()
                hvj[v]=self.sym_to_rv[v].range()[1]
            if len(solutions)>1 and not sympy.im(uj.subs(hvj))==0:
                continue
            uj_symbols = list(sorted(uj.atoms(sympy.Symbol), key = str))
            inv_transf = my_lambdify(uj_symbols, uj, "numpy")
            inv_transf_vars = [self.sym_to_rv[s] for s in uj_symbols]

            if params.models.debug_info:
                #print "vars to change: ", free_var.getSymname(), " <- ", dep_var.getSymname(), "=", self.rv_to_equation[dep_var]
                #print "equation: ", dep_var.getSymname(), "=", self.rv_to_equation[dep_var]
                print("substitution: ", free_var.getSymname(), "=", uj, end=' ')
                #print "variables: ", uj_symbols, inv_transf_vars

            # Jacobian
            #J = sympy.Abs(sympy.diff(uj, dep_var.getSymname()))
            J = sympy.diff(uj, dep_var.getSymname())
            print(J.atoms())
            J_symbols = list(sorted(J.atoms(sympy.Symbol), key = str))
            if len(J_symbols) > 0:
                jacobian_vars = [self.sym_to_rv[s] for s in J_symbols]
                jacobian = my_lambdify(J_symbols, J, "numpy")
                jacobian = NDFun(len(jacobian_vars), jacobian_vars, jacobian, safe = True, abs = True)
            else:
                jacobian = NDConstFactor(abs(float(J)))
                jacobian_vars = []

            if params.models.debug_info:
                print(";  Jacobian=", J)
            #print "variables: ", J_symbols, jacobian_vars

            var_changes.append((uj, inv_transf, inv_transf_vars, jacobian))
        return var_changes, equation

    def subst_for_rv_in_children(self, var, Xsym):
        """Substitute Xsym for occurrences of var in its children"""
        for rv, eq in self.rv_to_equation.items():
            if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                self.rv_to_equation[rv] = eq.subs(var.getSymname(), Xsym)

    def eliminate(self, var):
        var = self.prepare_var(var)
        if params.models.debug_info:
            print("eliminate variable: ", var.getSymname(), end=' ')
        if var in self.free_rvs:
            if params.models.debug_info:
                print(" eliminate free via integration")
            for rv, eq in self.rv_to_equation.items():
                if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                    raise RuntimeError("Cannot eliminate free variable on which other variables depend")
            self.nddistr = self.nddistr.eliminate(var)
            self.free_rvs.remove(var)
            self.all_vars.remove(var)
        elif var in self.dep_rvs:
            if params.models.debug_info:
                print(" eliminate dependent via substitution")
            subs_eq = self.rv_to_equation[var]
            del self.rv_to_equation[var]
            self.subst_for_rv_in_children(var, subs_eq)
            self.dep_rvs.remove(var)
            self.all_vars.remove(var)
        else:
            assert False
    def eliminate_other(self, vars):
        vars_to_eliminate = list(set(self.dep_rvs) - set(vars))
        #print "eliminate variables: ", ", ".join(str(rv.getSymname()) for rv in vars_to_eliminate)
        for var in vars_to_eliminate:
            self.eliminate(var)
    def unchain(self, vars, excluded=[]):
        vars_to_unchain = set(vars) - set(self.free_rvs)
        print("unchain variables: ", ", ".join(str(rv.getSymname()) for rv in vars_to_unchain))
        print("unchain variables: ",self.are_free(vars))
        print("unchain variables: ",self.are_free(vars_to_unchain))
        print(": ", vars_to_unchain)
        for i in range(len(vars_to_unchain)):
            for var in vars_to_unchain:
                #print ">>>>>>.", var.getSymname(), self.rv_to_equation[var]
                #print ">>", self.rv_to_equation[var].atoms
                if self.is_dependent(var) and self.are_free(self.rv_to_equation[var].atoms()):
                    for a in self.rv_to_equation[var].atoms():
                        print(">", a)
                        av = self.prepare_var(a)
                        print(">=", av)
                        print(self.__str__())
                        if self.is_free(av) and not av in set(excluded):
                            print("varschangeL:", av.getSymname(), var.getSymname())
                            self.varschange(av, var)
                            break

    def inference_to_remove(self, vars, condvars, condvals):
        assert len(condvars)==len(condvals), "condvars, condvals must be equal size"
        self.eliminate_other(set(vars) | set(condvars))
        self.unchain(set(vars) | set(condvars), excluded=vars)
        for i in range(len(condvars)):
            self.condition(condvars[i], condvals[i])
        for var in set(self.dep_rvs):
            self.eliminate(var)
        for var in set(self.free_rvs) - set(vars):
            self.eliminate(var)

    def inference(self, wanted_rvs, cond_rvs = [], cond_X = []):
        M = self.copy()
        wanted_rvs = set(wanted_rvs)
        cond = {}
        for v, x in zip(cond_rvs, cond_X):
            cond[v] = x
        ii=0
        while wanted_rvs != set(M.all_vars):
            print("OUTER LOOP| wanted:", [tmp_rv.getSymname() for tmp_rv in wanted_rvs], "all:", [tmp_rv.getSymname() for tmp_rv in M.all_vars], "dep:", [tmp_rv.getSymname() for tmp_rv in M.dep_rvs])
            # eliminate all dangling variables
            to_remove = []
            for v in M.dep_rvs:
                if v not in wanted_rvs and v not in cond and len(M.get_children(v)) == 0:
                    to_remove.append(v)
            for v in to_remove:
                M.eliminate(v)
            if len(to_remove) > 0:
                continue
            # propagate constants
            to_remove = []
            for v in M.dep_rvs:
                if len(M.get_parents(v)) == 0:
                    to_remove.append(v)
            for v in to_remove:
                if v not in wanted_rvs:
                    M.eliminate(v)
                else:
                    M.subst_for_rv_in_children(v, M.rv_to_equation[v])
            # a single itertion below reverses the DAG
            exchanged_vars = set()

            while wanted_rvs | exchanged_vars != set(M.all_vars):
                print("INNER LOOP| wanted:", [tmp_rv.getSymname() for tmp_rv in wanted_rvs], "exchanged:", [tmp_rv.getSymname() for tmp_rv in exchanged_vars], "all:", [tmp_rv.getSymname() for tmp_rv in M.all_vars])
                #print M.nddistr
                # find a free var to eliminate
                ii += 1
                if params.models.debug_info:
                    print("---------------step:", ii)
                    M.toGraphwiz(f =open("file"+str(ii)+".dot","w+"))
                    print(M)
                    print("---", ii, " ---> #free_vars:", len(M.free_rvs), "#dep_vars:", len(M.dep_rvs), "#eqns=", len(list(M.rv_to_equation.keys())), "sum=", (len(M.free_rvs) + len(M.dep_rvs)+len(list(M.rv_to_equation.keys()))))
                    print("------> #free_vars:", len(wanted_rvs), "#dep_vars:", len(exchanged_vars))
                    if have_pgv:
                        G = pgv.AGraph("file"+str(ii)+".dot")
                        G.layout("dot")
                        G.draw("file"+str(ii)+".pdf","pdf")
                to_remove = []
                for v in M.free_rvs:
                    if v not in wanted_rvs:
                        if v in cond or len(M.get_children(v)) == 0:
                            to_remove.append(v)
                # TODO: eliminate all vars at once so that NDProductDistr heuristic is used
                for v in to_remove:
                    if v not in cond:
                        M.eliminate(v)
                    else:
                        M.condition(v, cond[v])
                if len(to_remove) > 0:
                    continue
                # check if exchanging vars can bring any benefits at all
                dep_rvs_and_anc = set(M.dep_rvs) - exchanged_vars
                for rv in set(M.dep_rvs) - exchanged_vars:
                    dep_rvs_and_anc.update(M.get_parents(rv))
                if dep_rvs_and_anc.issubset(wanted_rvs):
                    break
                # find an unwanted free var and a dep var to exchange
                exchangeable_dep_vars = []
                for v in M.dep_rvs:
                    if v not in exchanged_vars and set(M.get_parents(v)).issubset(M.free_rvs):
                        exchangeable_dep_vars.append(v)
                pairs = []
                for dv in exchangeable_dep_vars:
                    nparents = len(M.get_parents(dv))
                    for fv in M.get_parents(dv):
                        nchildren = len(M.get_children(fv))
                        nterms = 0#M.nddistr.get_n_terms(fv) # TODO!!!
                        key = (1*(fv in wanted_rvs), (nparents-1)*(nchildren-1)) # heuristic for deciding which vars to exchange
                        #key = ((nparents-1 + nterms)*(nchildren-1), 1*(fv in wanted_rvs)) # heuristic for deciding which vars to exchange
                        pairs.append((key, fv, dv))
                print([(key, fv.getSymname(), dv.getSymname()) for key, fv, dv
                           in sorted(pairs, key=lambda x: (x[0], id(x[1]), id(x[2])))])
                if len(pairs) > 0:
                    pairs.sort(key=lambda x: (x[0], id(x[1]), id(x[2])))
                    _key, fv, dv = pairs[0]
                    M.varschange(fv, dv)
                    if fv not in wanted_rvs:
                        M.eliminate(fv)
                    else:
                        exchanged_vars.add(fv)
                else:
                    # whole graph has been reversed, may need to do it again in the outer loop...
                    break
        if params.models.debug_info:
            ii += 1
            print("---===-=-=-===-=--=", ii)
            print("==",M.toGraphwiz(f =open("file"+str(ii)+".dot","w+")))
            print(M)
            print("---", ii, " ---> #free_vars:", len(M.free_rvs), "#dep_vars:", len(M.dep_rvs), "#eqns=", len(list(M.rv_to_equation.keys())), "sum=", (len(M.free_rvs) + len(M.dep_rvs)+len(list(M.rv_to_equation.keys()))))
            print("------> #free_vars:", len(wanted_rvs), "#dep_vars:", len(exchanged_vars))
            if have_pgv:
                G = pgv.AGraph("file"+str(ii)+".dot")
                G.layout("dot")
                G.draw("file"+str(ii)+".pdf","pdf")
        return M
    def are_free(self, vars):
        for v in vars:
            if not self.is_free(v): return False
            return True
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
            for rv, eq in self.rv_to_equation.items():
                if var.getSymname() in set(eq.atoms(sympy.Symbol)):
                    note += 1
        return note

    def condition(self, var, X, **kwargs):
        var = self.prepare_var(var)
        if params.models.debug_info:
            print("condition on variable: ",  var.getSymname(), "=" ,X)
        if not self.is_free(var):
            raise RuntimeError("You can only condition on free variables")
        self.subst_for_rv_in_children(var, sympy.S(X))
        self.free_rvs.remove(var)
        self.all_vars.remove(var)
        self.nddistr = self.nddistr.condition([var], X)

    def as1DDistr(self):
        if len(self.dep_rvs) > 0:
            raise RuntimeError("Cannot get distribution of dependent variable.")
        if len(self.free_rvs) != 1:
            raise RuntimeError("Too many free variables")
        pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
        return pfun

    def as_const(self):
        if len(self.dep_rvs) == 1:
            return float(self.rv_to_equation[self.dep_rvs[0]])
        raise RuntimeError("unimplemented")

    def summary(self):
        if len(self.free_rvs) == 1:
            pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
            pfun.summary()
        else:
            raise RuntimeError("Too many variables.")

    def plot(self, **kwargs):
        if len(self.all_vars) == 1 and len(self.free_rvs) == 1:
            pfun = FunDistr(self.nddistr, breakPoints = self.nddistr.Vars[0].range())
            pfun.plot(label = str(self.nddistr.Vars[0].getSymname()), **kwargs)
            legend()
        elif len(self.all_vars) == 2 and len(self.free_rvs) == 2:
            plot_2d_distr(self.nddistr, **kwargs)
        elif len(self.all_vars) == 2 and len(self.free_rvs) == 1:
            a, b = self.free_rvs[0].range()
            freesym = self.free_rvs[0].getSymname()
            fun = my_lambdify([freesym], self.rv_to_equation[self.dep_rvs[0]], "numpy")
            ax = plot_1d1d_distr(self.nddistr, a, b, fun)
            ax.set_xlabel(self.free_rvs[0].getSymname())
            ax.set_ylabel(self.dep_rvs[0].getSymname())
        elif len(self.all_vars) == 1 and len(self.free_rvs) == 0:
            DiracSegment(self.as_const(), 1).plot(**kwargs)
        else:
            raise RuntimeError("Too many variables.")
    def is_free(self, var):
        var = self.prepare_var(var)
        return var in self.free_rvs

    def is_dependent(self, var):
        var = self.prepare_var(var)
        return var in self.dep_rvs

    def toGraphwiz(self, f = sys.stdout):
        print("digraph G {rankdir = BT", file=f)
        for key in self.free_rvs:
            print("\"{0}\"".format(key.getSymname()), " [label=\"{0}\"]".format(key.getSymname()), file=f)
        for key in self.dep_rvs:
            print("\"{0}\"".format(key.getSymname()), " [label=\"{0}={1}\"]".format(key.getSymname(),self.rv_to_equation[key]), file=f)
        for rv, eq in self.rv_to_equation.items():
            for a in eq.atoms(sympy.Symbol):
                print("\"{0}\" -> \"{1}\"".format(str(a), str(rv.getSymname())), file=f)
        print("}", file=f)

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
        #print "=====", self.vars
        #print self.symvars
        #print self.dep_rvs
        #print self.rv_to_equation
        self.symop = self.rv_to_equation[d]

        if len(self.vars) != 2:
            raise Exception("use it with two variables")
        x = self.symvars[0]
        y = self.symvars[1]
        z = sympy.Symbol("z")
        self.fun_alongx = eq_solve(self.symop, z, y)[0]
        self.fun_alongy = eq_solve(self.symop, z, x)[0]

        self.lfun_alongx = my_lambdify([x, z], self.fun_alongx, "numpy")
        self.lfun_alongy = my_lambdify([y, z], self.fun_alongy, "numpy")
        self.Jx = 1 * sympy.diff(self.fun_alongx, z)
        #print "Jx=", self.Jx
        #print "fun_alongx=", self.fun_alongx
        self.Jy = 1 * sympy.diff(self.fun_alongy, z)
        self.lJx = my_lambdify([x, z], self.Jx, "numpy")
        self.lJy = my_lambdify([y, z], self.Jy, "numpy")
        self.z = z
    def solveCutsX(self, fun, ay, by):
        axc = eq_solve(fun, ay, self.symvars[0])[0]
        bxc = eq_solve(fun, by, self.symvars[0])[0]
        return (axc, bxc)
    def solveCutsY(self, fun, ax, bx):
        #ayc = eq_solve(fun - ay, self.z)[0]
        #byc = eq_solve(fun - by, self.z)[0]
        ayc = fun.subs(self.symvars[0], ax)
        byc = fun.subs(self.symvars[0], bx)
        return (ayc, byc)
    def getUL(self, ax, bx, ay, by, z):
        axcz, bxcz = self.solveCutsX(self.fun_alongx, ay, by)
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
        figure()
        h = 0.1
        axis((ax - h, bx + h, ay - h, by + h))
        #print self.symvars
        #print self.d.getSym()
        #print self.symop
        x = self.symvars[0]
        y = self.symvars[1]

        lop = my_lambdify([x, y], self.symop, "numpy")
        tmp = [lop(ax, ay), lop(ax, by), lop(bx, ay), lop(bx, by)]
        i0, i1 = min(tmp), max(tmp)
        for i in linspace(i0, i1, 20):
            #y =  self.lfun_alongx(t, i)
            #plot(t, y)
            try:
                L, U = self.getUL(ax, bx, ay, by, i)
                tt = linspace(L, U, 100)
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
        plot()
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
        lop = my_lambdify([x, y], op, "numpy")
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
                    pass
                    #print "not a number, xi=", xi, "yi=", yi, "result=", lop(xi,yi)
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
        op = self.symop #d.getSym()
        x = self.symvars[0]
        y = self.symvars[1]
        lop = sympy.lambdify([x, y], op, "numpy")
        if size(xx) == 1:
            xx = asfarray([xx])
        wyn = zeros_like(xx)
        P = self.nddistr
        #fun = lambda t : P.cdf(t, self.lfun_alongx(t, array([zj]))) * abs(self.lJx(t, array([zj])))
#        if isinstance(P, pacal.depvars.copulas.MCopula) | isinstance(P, pacal.depvars.copulas.WCopula):
#            #print ">>", P
#            #funPdf = lambda t : P.cdf(t, self.lfun_alongx(t, zj)) * abs(self.lJx(t, zj))
#            funCdf = lambda t : P.cdf(t, self.lfun_alongx(t, zj)) #* abs(self.lJx(t, zj))
#            fun = funCdf
        #if isinstance(P, pacal.depvars.copulas.PiCopula):
        #    print "Piiii"
        #    fun = lambda t : segi(t) * segj(self.lfun_alongx(t, array([zj]))) * abs(self.lJx(t, array([zj])))
        #else:
        fun = lambda t : P.pdf(t, self.lfun_alongx(t, zj)) * abs(self.lJx(t, zj))
        ##fun = lambda t : P.jpdf_(F, G, t, self.lfun_alongx(t, array([zj])))  * abs(self.lJx(t, array([zj])))


        for j in range(len(xx)) :
            zj = xx[j]
#            if isinstance(P, pacal.depvars.copulas.WCopula):
#                I = 1
#            else:
            I = 0
            err = 0
            for segi, segj in segList:
                #print segi, segj
                if segi.isSegment() and segj.isSegment():
                    L, U = self.getUL(segi.a, segi.b, segj.a, segj.b, zj)
                    L, U  = min(U, L), max(U, L)
                    if L < U:
                        i, e = self._segint(fun, float(L), float(U), debug_info=False)
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
#                if isinstance(P, pacal.depvars.copulas.WCopula):
#                    I = min(I,i)
#                elif isinstance(P, pacal.depvars.copulas.MCopula):
#                    I = max(I,i)
#                else:
                I += i
                #print I
                err += e
            wyn[j] = I
        return wyn

    def eval(self):
        return PDistr(self.convmodel())
    def varchange_and_eliminate(self):
        return self.eval()

def _findSegList(f, g, z, op):
    """It find list of segments for integration purposes, for given z
    input: f, g - piecewise function, z = op(x,y), op - operation (+ - * /)
    output: list of segment products depends on z
    """
    slist = []
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
    from pacal.depvars.copulas import GumbelCopula, GumbelCopula2d, ClaytonCopula, FrankCopula2d, FrankCopula
    from pacal.depvars.copulas import PiCopula, WCopula, MCopula


    X = UniformDistr(1, 2, sym="X")
    Y = UniformDistr(1, 3, sym="Y")

    S = X + Y; S.setSym("S")
    #P = NDProductDistr([X, Y])
    M = Model([X, Y], [S])
    print(M)
    #M2 = M.inference(wanted_rvs = [X])
    #M2 = M.inference(wanted_rvs = [X], cond_rvs = [Y], cond_X = [1.5])
    #M2 = M.inference(wanted_rvs = [S])
    #M2 = M.inference(wanted_rvs = [S], cond_rvs = [Y], cond_X = [1.5]) #! NaN moments!
    #M2 = M.inference(wanted_rvs = [X], cond_rvs = [S], cond_X = [2.5])
    M2 = M.inference(wanted_rvs = [X, Y], cond_rvs = [S], cond_X = [2.5]).plot()
    print("===", M2)

    #M.plot()
    show()
    0/0

    figure()
    N = X * Y; N.setSym("N")
    D = X + Y; D.setSym("D")
    R = N / D; R.setSym("R")
    P = NDProductDistr([X, Y])
    M = Model(P, [N, D, R])
    print(M)
    M2 = M.inference(wanted_rvs = [R])
    M2.plot()
    0/0
    M.varschange(X, N)
    M.eliminate(X)
    print(M)
    M.plot()
    show()
    0/0

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
#    funw = Mw.eval()
#    funm = Mm.eval()
#    #funp = Mp.eval()
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
#        funi = Mi.eval()
#        funi.get_piecewise_cdf().plot(color="g")
#        funi.summary()
#    for theta in [-15, -5, 5, 15]:
#        print "::::", theta
#        ci = FrankCopula2d(marginals=[X, Y], theta=theta)
#        Mi = TwoVarsModel(ci, U)
#        funi = Mi.eval()
#        funi.get_piecewise_cdf().plot(color="b")
#        funi.summary()
#    for theta in [5, 10]:
#        print ":::", theta
#        ci = ClaytonCopula(marginals=[X, Y], theta=theta)
#        Mi = TwoVarsModel(ci, U)
#        funi = Mi.eval()
#        funi.get_piecewise_cdf().plot(color="r")
#        funi.summary()
#    print "==============="
#    V= X-Y
#    cp = PiCopula(marginals=[X, Y])
#    m = TwoVarsModel(cp, V)
#    fun = m.eval()
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

    print("p=", V.parents[1].getSym())
    mR = TwoVarsModel(cij, V)
    funR = mR.eval()
    funR.summary()
    funR.plot(color="k")

#    cc = ClaytonCopula(marginals=[X1, X2], theta=1.0/10.0)
#    cc.plot()
#    mC = TwoVarsModel(cc,V)
#    funC = mC.eval()
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
