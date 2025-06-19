from problem_variables import ALMVar, ALMParameters, SSNParameters
from prox import *

def update_tol(SP: SSNParameters, t: int):
    if t < 2:
        SP.tol_ssn = 1e-2
    elif 2 <= t <= 10:
        SP.tol_ssn = 1e-4
    else:
        SP.tol_ssn = 1e-6

def update_iter(SP: SSNParameters, t: int):
    SP.max_iter_ssn = 10 if t <= 2 else 25

def update_sigma_gamma(almvar: ALMVar, AP: ALMParameters):
    almvar.sigma *= AP.sigma_scale
    almvar.gamma = 1.0 / almvar.sigma

def update_proxmethod(almvar: ALMVar):
    gamma = almvar.gamma
    if gamma < 2:
        almvar.prox_method_ls = prox_smallgamma_ls
        almvar.prox_method_ssn = prox_smallgamma_ssn
    elif gamma == 2:
        almvar.prox_method_ls = prox_gamma2_ls
        almvar.prox_method_ssn = prox_gamma2_ssn
    else:
        almvar.prox_method_ls = prox_largegamma_ls
        almvar.prox_method_ssn = prox_largegamma_ssn

def update_proxmethod_thread(almvar: ALMVar):
    gamma = almvar.gamma
    if gamma < 2:
        almvar.prox_method_ls = prox_smallgamma_ls
        almvar.prox_method_ssn = prox_smallgamma_ssn
    elif gamma == 2:
        almvar.prox_method_ls = prox_gamma2_ls
        almvar.prox_method_ssn = prox_gamma2_ssn
    else:
        almvar.prox_method_ls = prox_largegamma_ls
        almvar.prox_method_ssn = prox_largegamma_ssn
