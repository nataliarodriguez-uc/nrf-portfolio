import pandas as pd
import numpy as np

def summarize_almlog(almlog, dataset_name):
    """Summarize ALM log results into a single-row DataFrame."""

    if almlog.alm_iter == 0:
        ssn_iters_used = []
    else:
        ssn_iters_used = almlog.ssn_iters[:almlog.alm_iter]

    total_ssn_iters = sum(ssn_iters_used)
    total_lsearch_iters = sum(sum(inner) for inner in almlog.lsearch_iters[:almlog.alm_iter])

    ssn_times = almlog.ssn_times[:almlog.alm_iter]
    ssn_wD_times = almlog.ssn_wD_times[:almlog.alm_iter]
    prox_times_flat = np.concatenate(almlog.prox_times[:almlog.alm_iter]) if almlog.alm_iter > 0 else []
    lsearch_times_flat = np.concatenate(almlog.lsearch_times[:almlog.alm_iter]) if almlog.alm_iter > 0 else []

    summary = {
        "dataset": dataset_name,
        "L_final": almlog.L_final,
        "alm_time": almlog.alm_time,
        "alm_iter": almlog.alm_iter,

        "ssn_time_total": sum(ssn_times),
        "ssn_time_avg": np.mean(ssn_times) if ssn_times else 0.0,
        "ssn_iters_total": total_ssn_iters,
        "ssn_iters_avg": np.mean(ssn_iters_used) if ssn_iters_used else 0.0,

        "ssn_wD_time_total": sum(ssn_wD_times),
        "ssn_wD_time_avg": np.mean(ssn_wD_times) if ssn_wD_times else 0.0,

        "prox_time_total": sum(prox_times_flat) if len(prox_times_flat) else 0.0,
        "prox_time_avg": np.mean(prox_times_flat) if len(prox_times_flat) else 0.0,

        "prox_allocs_total": sum(almlog.prox_allocs[:almlog.alm_iter]),
        "prox_allocs_avg": np.mean(almlog.prox_allocs[:almlog.alm_iter]) if almlog.alm_iter > 0 else 0.0,

        "lsearch_time_total": sum(lsearch_times_flat) if len(lsearch_times_flat) else 0.0,
        "lsearch_time_avg": np.mean(lsearch_times_flat) if len(lsearch_times_flat) else 0.0,

        "lsearch_iters_total": total_lsearch_iters,
        "lsearch_iters_avg": total_lsearch_iters / total_ssn_iters if total_ssn_iters > 0 else 0.0,

        "lsearch_allocs_total": sum(almlog.lsearch_allocs[:almlog.alm_iter]),
        "lsearch_allocs_avg": np.mean(almlog.lsearch_allocs[:almlog.alm_iter]) if almlog.alm_iter > 0 else 0.0
    }

    df = pd.DataFrame([summary])
    df.to_csv(f"almlog_{dataset_name}.csv", index=False)
    return df

def read_all_almlog_csvs(prefix, N, export_path=None):
    """Read multiple ALM log CSVs and optionally save the combined DataFrame."""
    dfs = []

    for i in range(1, N+1):
        filename = f"{prefix}_dataset{i}.csv"
        df = pd.read_csv(filename)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    if export_path is not None:
        full_df.to_csv(export_path, index=False)

    return full_df

def export_w_alm_from_PI(PI, name, sigma0, tau0, alpha0, AP, SP, LS):
    print(f"Solving and exporting ALM w for {name} ...")
    almvar, almlog = alm(sigma0, tau0, alpha0, PI, AP, SP, LS)

    # Export weights
    np.savetxt(f"{name}_w.csv", almvar.w, delimiter=",")

    # Summarize and export logs
    df = summarize_almlog(almlog, name)
    df.to_csv(f"almlog_{name}.csv", index=False)

    # Angle comparisons
    if hasattr(PI, 'w_svm'):
        angle_svm = degrees(arccos(dot(PI.w_svm / norm(PI.w_svm), almvar.w / norm(almvar.w))))
        print(f"Angle between w_svm and w*: {angle_svm:.2f}°")

    if hasattr(PI, 'w0_shifted'):
        angle_shifted = degrees(arccos(dot(PI.w0_shifted / norm(PI.w0_shifted), almvar.w / norm(almvar.w))))
        print(f"Angle between w0_shifted and w*: {angle_shifted:.2f}°")
    
    return almvar, almlog