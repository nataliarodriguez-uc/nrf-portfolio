import numpy as np
from prox import compute_prox_ls

def compute_line_search(t, k, almlog, almvar, ssnvar, proxvar, d, PI, LS):
    # Initialize alpha
    alpha_ls = ssnvar.alpha_ssn

    # Objective and gradient at current point
    L_current = proxvar.Lag_obj
    J_current = proxvar.Lag_J

    for l in range(LS.max_iter_ls):
        
        # (1) Candidate primal and D-product update
        ssnvar.w_ls = ssnvar.w_ssn + alpha_ls * d
        proxvar.w_ls_D = ssnvar.w_ssn_D + alpha_ls * proxvar.d_D  # linear update

        # (2) Compute new objective value
        L_new = compute_prox_ls(proxvar.w_ls_D, almvar, proxvar, PI)

        # (3) Armijo condition
        if L_new - L_current <= LS.c * alpha_ls * np.dot(J_current, d):
            almlog.lsearch_iters[t][k] = l
            break
        else:
            alpha_ls *= LS.beta

    # Update step size
    ssnvar.alpha_ssn = alpha_ls
