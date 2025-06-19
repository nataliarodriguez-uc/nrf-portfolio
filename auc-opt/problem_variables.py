import numpy as np

class ALMVar:
    def __init__(self, tau_init, sigma_init, PI):
        self.cons_condition_norm = []
        self.cons_condition = np.zeros(len(PI.K))
        self.lambd = np.zeros(len(PI.K))
        self.tau = tau_init
        self.sigma = sigma_init
        self.gamma = 1.0 / sigma_init
        self.w = np.random.randn(PI.n)
        self.y = np.zeros(len(PI.K))
        self.alpha = 1.0
        self.delta = 0.0
        self.D_product = [np.outer(PI.D[:, k], PI.D[:, k]) for k in range(len(PI.K))]
        self.w_D = np.zeros(len(PI.K))
        self.prox_method_ssn = None  # to be assigned
        self.prox_method_ls = None


class ALMParameters:
    def __init__(self, max_iter_alm, tau_scale, sigma_scale, tol_alm):
        self.max_iter_alm = max_iter_alm
        self.tau_scale = tau_scale
        self.sigma_scale = sigma_scale
        self.tol_alm = tol_alm


class ProxVar:
    def __init__(self, n, K, tau):
        self.y = np.zeros(K)
        self.Lprox = np.zeros(K)
        self.Jprox = np.zeros((n, K))
        self.Hprox = [np.zeros((n, n)) for _ in range(K)]
        self.Hprox_diag = tau * np.eye(n)
        self.w_ls_D = np.zeros(K)
        self.d_D = np.zeros(K)
        self.w_Dij = np.zeros(K)
        self.Lag_obj = 0.0
        self.Lag_J = np.zeros(n)
        self.Lag_H = np.zeros((n, n))
        self.prox_time = 0.0


class LineSearchParameters:
    def __init__(self, c, max_iter_ls, beta):
        self.c = c
        self.max_iter_ls = max_iter_ls
        self.beta = beta


class SSNVar:
    def __init__(self, PI):
        self.L_obj = 0.0
        self.L_grad = np.zeros(PI.n)
        self.L_hess = np.zeros((PI.n, PI.n))
        self.alpha_ssn = 1.0
        self.w_ssn = np.zeros(PI.n)
        self.w_ls = np.zeros(PI.n)
        self.y_ssn = np.zeros(len(PI.K))
        self.w_ssn_D = np.zeros(len(PI.K))


class SSNParameters:
    def __init__(self, tol_ssn, max_iter_ssn):
        self.tol_ssn = tol_ssn
        self.max_iter_ssn = max_iter_ssn


class ALMLog:
    def __init__(self, max_alm, max_ssn):
        self.alm_time = 0.0
        self.alm_iter = 0
        self.L_final = 0.0
        self.ssn_times = np.zeros(max_alm)
        self.ssn_iters = np.zeros(max_alm, dtype=int)
        self.ssn_wD_times = np.zeros(max_alm)
        self.ssn_d_times = [np.zeros(max_ssn) for _ in range(max_alm)]
        self.prox_times = [np.zeros(max_ssn) for _ in range(max_alm)]
        self.prox_allocs = np.zeros(max_alm)
        self.lsearch_times = [np.zeros(max_ssn) for _ in range(max_alm)]
        self.lsearch_iters = [np.zeros(max_ssn, dtype=int) for _ in range(max_alm)]
        self.lsearch_allocs = np.zeros(max_alm)
