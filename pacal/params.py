from inspect import getmembers

def _str_params_list(p, depth = 0):
    """Recursively print parameters"""
    slist = ["    " * depth + "==" + p.__name__ + "=="]
    #for par, v in sorted(p.__dict__.iteritems()):
    for par, v in getmembers(p):
        if par[0] == '_' or par in ["str", "str_params", "finfo", "double", "params_class", "getmembers"]:
            continue
        if hasattr(v, "__dict__"):
            slist += _str_params_list(v, depth + 1)
        else:
            slist.append("    " * (depth+1) + str(par) + ": " + str(v))
    return slist
def str_params(p = None):
    if p is None:
        from . import params
        p = params
    slist = _str_params_list(p)
    return "\n".join(slist)

class params_class(object):
    """Base class for sets of parameters."""
    @classmethod
    def str(c):
        return str_params(c)

###########################
#### global parameters ####
###########################
import os

class general(params_class):
    warn_on_dependent = True
    class distr(params_class):
        independent = True
    parallel = (os.name == 'posix')
    nprocs = None
    process_pool = None

# disable threading in openblas when parallel computing is used this
# is an awkward place to put this code but must be done before numpy
# import
if general.parallel:
    try:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except:
        print("WARNING: could not disable openblas threading")


from numpy import finfo, double


class pole_detection(params_class):
    max_pole_exponent = -1e-2 # exponents above this value are treated as poles
    min_distance_from_pole = 1e-2 # integration bounds this close to a
                                  # pole will be integrated using
                                  # integration with poles
    continuity_eps = 1e2 * finfo(double).eps # consider function jumps
                                             # smaller than this as
                                             # continuous
    derivative = True   # whether derivative should
                        # be checked in testPole
                        # generally it works better with True
                        # but for some cases (like product of beta)
                        # one can try to set False

# default convergence test
class convergence(params_class):
    abstol = 16 * finfo(double).eps
    reltol = 16 * finfo(double).eps
    min_quit_iter = 10000 # quit no earlier than this iteration
    min_quit_no_improvement = 2 # quit if no improvement for this many
                                # steps
    min_improvement_ratio = 0.5 # minimum decrease in error considered
                                # an improvement
    force_nonzero = False # stop only when targer value is nonzero
                          # helps avoiding too sparse grids missing
                          # all nonzero points

# default interpolation parameters
class interpolation(params_class):
    maxn = 100
    debug_info = False
    debug_plot = False
    use_cheb_2nd = True    # always use interpolator based on
                            # chebyshev nodes of 2nd kind (faster and accurate at ends of intervals)
                            # if False use interpolator based on nodes
                            # of 1st kind (no nodes at ends of intervals)
    class convergence(convergence):
        abstol = 4 * finfo(double).eps
        reltol = 4 * finfo(double).eps
# default integration parameters
class integration(params_class):
    maxn = 1000
    debug_info = False
    debug_plot = False
    wide_condition = 1e1
    class convergence(convergence): pass


###################################################
#### parameters for specific types of segments ####
###################################################


# interpolation on finite/infinite/asymptotic/pole segments
class interpolation_finite(interpolation): pass
class interpolation_infinite(interpolation):
    maxn = 100
    exponent = 6
class interpolation_asymp(interpolation):
    maxn = 100
    class convergence(convergence):
        abstol = 16 * finfo(double).eps
        reltol = 16 * finfo(double).eps
class interpolation_pole(interpolation):
    maxn = 200

class interpolation_nd(interpolation):
    maxq = 7
    class convergence(interpolation.convergence):
        abstol = 1e-8
        reltol = 1e-8
        force_nonzero = True
    debug_info = True
    debug_plot = False

# integration in arithmetic operations for target value in
# finite/infinite/asymptotic/pole segments
class integration_finite(integration):
    debug_plot = True
    maxn = 10000
class integration_infinite(integration):
    exponent = 6
class integration_asymp(integration):
    maxn = 1000
class integration_pole(integration):
    exponent = 8
    maxn = 1000

class segments(params_class):
    abstol = 1e-16
    reltol = 1e-16
    debug_info = False
    debug_plot = False
    unique_eps = 1e-14 # collapse segments shorter than this value
    class plot(params_class):
        numberOfPoints  = 1000
        leftRightEpsilon  = 1e-20
        yminEpsilon  = 1e-3
        nodeMarkerSize  = 3
        ciLevel = None        # a confidence interval level for the range of plots
        showNodes = False
        showSegments = False
    class cumint(params_class):
        abstol = 1e-16
        reltol = 1e-16
        maxiter = 1000
    class inverseintegral(params_class):
        abstol = 1e-16
        reltol = 1e-16
    class integration(integration): pass
    class summary(params_class):
        identify = False  # if True it identify summary numbers using mpmath's identify function

class models(params_class):
    debug_info = False
    debug_plot = False

if __name__ == "__main__":
    print("integration.convergence.reltol=", integration.convergence.reltol)
    print(str_params())
    print()
    print(str_params(integration))
    print(segments.integration.maxn)
    print(interpolation_finite.maxn)
    print(interpolation_asymp.maxn)
