"""
Consolidated TEROS 12 + TEROS 21 restricted van Genuchten block-bootstrap analysis.

Required CSV columns:
    time_hours, suction_kpa, theta
Optional:
    sigma_theta

Rows MUST represent time-aligned TEROS 12 (theta) and TEROS 21 (suction)
measurements and MUST remain in chronological order.

Main outputs:
    residual ACF and Ljung-Box diagnostics
    stationary/circular/moving residual block bootstrap
    IID-bootstrap comparison
    alpha-n and alpha-n-m covariance/correlation matrices
    percentile parameter intervals
    SWCC, Kr and Ku pointwise percentile bands
    CSV/Excel outputs and PNG plots
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

try:
    from arch.bootstrap import optimal_block_length
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# ============================================================
# USER SETTINGS — EDIT THESE FOR YOUR SOIL/TEST
# ============================================================
THETA_R = 7.7        # Units: %
THETA_S = 15.1       # Units: %
INITIAL_ALPHA = 0.00339       # 1/kPa when suction is kPa
INITIAL_N = 1.6648
ALPHA_BOUNDS = (1.0e-8, 1.0e3)
N_BOUNDS = (1.001, 20.0)
KS = 6.871e-6                 # e.g. m/s
PORE_CONNECTIVITY_L = 0.5
RANDOM_SEED = 20260929
N_STARTS_ORIGINAL = 8
N_STARTS_BOOTSTRAP = 3
MAX_NFEV = 5000
MAX_BOOTSTRAP_ATTEMPT_FACTOR = 1.5
USE_ABSOLUTE_SUCTION = True # converts negative TEROS 21 water potential to |psi|
THETA_IS_PERCENT = False    # True only if theta is 32 rather than 0.32
N_GRID = 250


def vg_theta(psi, alpha, n):
    """Restricted van Genuchten retention function, m=1-1/n."""
    psi = np.asarray(psi, dtype=float)
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(psi, np.finfo(float).tiny)
    log_x = np.log(alpha) + np.log(psi_safe)
    log_term = np.logaddexp(0.0, n * log_x)
    se = np.exp(-m * log_term)
    return THETA_R + (THETA_S - THETA_R) * se


def effective_saturation(psi, alpha, n):
    psi = np.asarray(psi, dtype=float)
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(psi, np.finfo(float).tiny)
    log_x = np.log(alpha) + np.log(psi_safe)
    log_term = np.logaddexp(0.0, n * log_x)
    se = np.exp(-m * log_term)
    return np.clip(se, 1e-14, 1.0 - 1e-14)


def vg_relative_conductivity(psi, alpha, n, l=0.5):
    """Mualem-van Genuchten relative hydraulic conductivity."""
    m = 1.0 - 1.0 / n
    se = effective_saturation(psi, alpha, n)
    inner = np.clip(1.0 - se ** (1.0 / m), 0.0, 1.0)
    kr = se ** l * (1.0 - inner ** m) ** 2
    return np.clip(kr, 0.0, 1.0)


def physical_to_transformed(alpha, n):
    if alpha <= 0 or n <= 1:
        raise ValueError("alpha must be >0 and n must be >1")
    return np.array([np.log(alpha), np.log(n - 1.0)], dtype=float)


def transformed_to_physical(z):
    alpha = float(np.exp(z[0]))
    n = float(1.0 + np.exp(z[1]))
    m = float(1.0 - 1.0 / n)
    return alpha, n, m


LOWER_Z = physical_to_transformed(ALPHA_BOUNDS[0], N_BOUNDS[0])
UPPER_Z = physical_to_transformed(ALPHA_BOUNDS[1], N_BOUNDS[1])


def make_template(path: Path):
    """Write a minimal input template."""
    pd.DataFrame({
        "time_hours": [0, 1, 2, 3, 4],
        "suction_kpa": [0.1, 0.15, 0.22, 0.35, 0.55],
        "theta": [0.420, 0.419, 0.417, 0.414, 0.410],
    }).to_csv(path, index=False)


def load_teros_data(path: Path):
    df = pd.read_csv(path)
    required = {"time_hours", "suction_kpa", "theta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    df = df.sort_values("time_hours").reset_index(drop=True)

    if USE_ABSOLUTE_SUCTION:
        df["suction_kpa"] = df["suction_kpa"].abs()

    if THETA_IS_PERCENT:
        df["theta"] = df["theta"] / 100.0
        if "sigma_theta" in df.columns:
            df["sigma_theta"] = df["sigma_theta"] / 100.0

    if (df["suction_kpa"] <= 0).any():
        pos = df.loc[df["suction_kpa"] > 0, "suction_kpa"]
        replacement = max(float(pos.min()) * 1e-6, 1e-12) if len(pos) else 1e-12
        df.loc[df["suction_kpa"] <= 0, "suction_kpa"] = replacement

    if "sigma_theta" not in df.columns:
        df["sigma_theta"] = 1.0
    if (df["sigma_theta"] <= 0).any():
        raise ValueError("sigma_theta must be positive")

    if df["time_hours"].duplicated().any():
        warnings.warn("Duplicate timestamps detected. Verify TEROS 12/21 pairing.")

    return df


def fit_restricted_vg(suction, theta_obs, sigma_theta, initial_z=None,
                      n_starts=3, rng=None):
    suction = np.asarray(suction, float)
    theta_obs = np.asarray(theta_obs, float)
    sigma_theta = np.asarray(sigma_theta, float)
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng

    if initial_z is None:
        initial_z = physical_to_transformed(INITIAL_ALPHA, INITIAL_N)
    initial_z = np.clip(initial_z, LOWER_Z + 1e-10, UPPER_Z - 1e-10)

    starts = [initial_z]
    for _ in range(n_starts - 1):
        trial = initial_z + rng.normal(0.0, [0.8, 0.35])
        starts.append(np.clip(trial, LOWER_Z + 1e-10, UPPER_Z - 1e-10))

    def residual_fun(z):
        alpha, n, _ = transformed_to_physical(z)
        return (vg_theta(suction, alpha, n) - theta_obs) / sigma_theta

    best = None
    for start in starts:
        try:
            res = least_squares(
                residual_fun, x0=start, bounds=(LOWER_Z, UPPER_Z),
                method="trf", jac="3-point", x_scale="jac", loss="linear",
                max_nfev=MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            valid = res.success and np.isfinite(res.cost) and np.all(np.isfinite(res.x))
            if valid and (best is None or res.cost < best.cost):
                best = res
        except Exception:
            pass

    if best is None:
        raise RuntimeError("Restricted VG nonlinear regression failed")

    alpha, n, m = transformed_to_physical(best.x)
    pred = vg_theta(suction, alpha, n)
    resid = theta_obs - pred
    sse = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    sst = float(np.sum((theta_obs - np.mean(theta_obs))**2))
    r2 = np.nan if sst == 0 else float(1.0 - sse/sst)

    return dict(alpha=alpha, n=n, m=m, z=best.x.copy(), predicted=pred,
                residuals=resid, sse=sse, rmse=rmse, r2=r2)


def residual_diagnostics(residuals, time_hours):
    residuals = np.asarray(residuals, float)
    time_hours = np.asarray(time_hours, float)
    n = len(residuals)
    max_lag = min(40, max(1, n//5))

    acf_val, acf_ci = acf(residuals, nlags=max_lag, fft=True, alpha=0.05)
    acf_df = pd.DataFrame({"lag": np.arange(len(acf_val)), "acf": acf_val,
                           "ci_lower": acf_ci[:,0], "ci_upper": acf_ci[:,1]})

    cand = [1,2,3,5,10,15,20,max_lag]
    lags = sorted(set(x for x in cand if 1 <= x <= max_lag))
    lb = acorr_ljungbox(residuals, lags=lags, model_df=0, return_df=True)
    lb.index.name = "lag"

    pos_sum = 0.0
    for rho in acf_val[1:]:
        if rho <= 0:
            break
        pos_sum += float(rho)
    neff = float(n / (1.0 + 2.0*pos_sum))

    dt = np.diff(time_hours)
    med_dt = float(np.median(dt)) if len(dt) else np.nan
    mean_dt = float(np.mean(dt)) if len(dt) else np.nan
    cv_dt = float(np.std(dt, ddof=1)/mean_dt) if len(dt)>1 and mean_dt>0 else np.nan

    return dict(acf=acf_df, ljung_box=lb, durbin_watson=float(durbin_watson(residuals)),
                effective_n_approx=neff, median_sampling_hours=med_dt,
                sampling_interval_cv=cv_dt)


def prepare_adjusted_residuals(raw_residuals, sigma_theta, p=2):
    std = np.asarray(raw_residuals,float)/np.asarray(sigma_theta,float)
    centered = std - np.mean(std)
    n = len(centered)
    if n <= p:
        raise ValueError("N must exceed number of fitted parameters")
    return np.sqrt(n/(n-p))*centered


def choose_block_length(residuals, method, override=None):
    residuals = np.asarray(residuals,float)
    n = len(residuals)
    if override is not None:
        return max(2, min(int(override), n))

    if ARCH_AVAILABLE:
        try:
            out = optimal_block_length(pd.Series(residuals, name="residual"))
            row = out.iloc[0]
            if method == "stationary":
                val = row["stationary"] if "stationary" in row.index else row.iloc[0]
            else:
                val = row["circular"] if "circular" in row.index else row.iloc[-1]
            if np.isfinite(val) and val >= 1:
                return int(max(2, min(round(float(val)), max(2,n//3))))
        except Exception as exc:
            warnings.warn(f"Automatic block length selection failed: {exc}")

    return max(2, min(int(np.ceil(n**(1/3))), max(2,n//3)))


def iid_resample(residuals, rng):
    return rng.choice(residuals, size=len(residuals), replace=True)


def moving_resample(residuals, L, rng):
    residuals = np.asarray(residuals); n=len(residuals); L=max(1,min(int(L),n)); out=[]
    while len(out)<n:
        start=int(rng.integers(0,n-L+1)); out.extend(residuals[start:start+L].tolist())
    return np.asarray(out[:n],float)


def circular_resample(residuals, L, rng):
    residuals=np.asarray(residuals); n=len(residuals); L=max(1,int(L)); out=[]
    while len(out)<n:
        start=int(rng.integers(0,n)); idx=(start+np.arange(L))%n; out.extend(residuals[idx].tolist())
    return np.asarray(out[:n],float)


def stationary_resample(residuals, mean_L, rng):
    residuals=np.asarray(residuals); n=len(residuals); L=max(1.0,float(mean_L)); p=1.0/L
    out=np.empty(n,float); idx=int(rng.integers(0,n))
    for i in range(n):
        out[i]=residuals[idx]
        idx=int(rng.integers(0,n)) if rng.random()<p else (idx+1)%n
    return out


def resample(residuals, method, L, rng):
    if method=="iid": return iid_resample(residuals,rng)
    if method=="stationary": return stationary_resample(residuals,L,rng)
    if method=="circular": return circular_resample(residuals,L,rng)
    if method=="moving": return moving_resample(residuals,L,rng)
    raise ValueError("method must be stationary, circular, moving or iid")


def run_bootstrap(suction, fitted_theta, adjusted_residuals, sigma_theta,
                  original_z, method, L, B, rng):
    records=[]; attempts=0; failed=0
    max_attempts=int(np.ceil(MAX_BOOTSTRAP_ATTEMPT_FACTOR*B))

    while len(records)<B and attempts<max_attempts:
        attempts += 1
        rstar = resample(adjusted_residuals, method, L, rng)
        theta_star = fitted_theta + sigma_theta*rstar
        try:
            fit = fit_restricted_vg(suction, theta_star, sigma_theta,
                                    initial_z=original_z,
                                    n_starts=N_STARTS_BOOTSTRAP, rng=rng)
            a, n = fit["alpha"], fit["n"]
            m = 1.0 - 1.0/n
            if not np.all(np.isfinite([a,n,m])):
                raise RuntimeError
            records.append({"replicate":len(records)+1,"alpha":a,"n":n,"m":m})
        except Exception:
            failed += 1

    if len(records)<B:
        raise RuntimeError(f"Only {len(records)} successful fits after {attempts} attempts")
    return pd.DataFrame(records), {"successful":len(records),"attempts":attempts,
                                   "failed":failed,"failure_rate":failed/attempts}


def parameter_summary(df, original):
    rows=[]
    originals={k:original[k] for k in ["alpha","n","m"]}
    for p in ["alpha","n","m"]:
        x=df[p].to_numpy(float); q=np.quantile(x,[0.025,0.5,0.975]); mean=x.mean()
        rows.append({"parameter":p,"original_estimate":originals[p],"bootstrap_mean":mean,
                     "bootstrap_bias":mean-originals[p],"bootstrap_SE":np.std(x,ddof=1),
                     "CI_2.5_percentile":q[0],"median":q[1],"CI_97.5_percentile":q[2],
                     "CI_width":q[2]-q[0]})
    return pd.DataFrame(rows)


def propagate_curves(df, psi_grid):
    B=len(df); M=len(psi_grid)
    T=np.empty((B,M)); KR=np.empty((B,M)); KU=np.empty((B,M))
    for j,row in enumerate(df.itertuples(index=False)):
        T[j]=vg_theta(psi_grid,row.alpha,row.n)
        KR[j]=vg_relative_conductivity(psi_grid,row.alpha,row.n,PORE_CONNECTIVITY_L)
        KU[j]=KS*KR[j]
    def q(a): return np.quantile(a,[0.025,0.5,0.975],axis=0)
    tq, krq, kuq=q(T),q(KR),q(KU)
    return pd.DataFrame({"suction_kpa":psi_grid,
                         "theta_2.5":tq[0],"theta_50":tq[1],"theta_97.5":tq[2],
                         "Kr_2.5":krq[0],"Kr_50":krq[1],"Kr_97.5":krq[2],
                         "Ku_2.5":kuq[0],"Ku_50":kuq[1],"Ku_97.5":kuq[2]})


def make_plots(outdir, time, std_resid, acf_df, suction, theta, boot, curves):
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(time,std_resid); ax.axhline(0,lw=1)
    ax.set(xlabel="Time (h)",ylabel="Standardized residual",title="TEROS VG residuals")
    fig.tight_layout(); fig.savefig(outdir/"residuals_vs_time.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4)); ax.stem(acf_df["lag"],acf_df["acf"])
    s=1.96/np.sqrt(len(std_resid)); ax.axhline(s,ls="--"); ax.axhline(-s,ls="--")
    ax.set(xlabel="Lag",ylabel="Autocorrelation",title="Residual ACF")
    fig.tight_layout(); fig.savefig(outdir/"residual_ACF.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(boot["alpha"],boot["n"],s=8,alpha=.35)
    ax.set(xlabel="alpha",ylabel="n",title="Joint block-bootstrap distribution")
    fig.tight_layout(); fig.savefig(outdir/"alpha_n_joint.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5)); ax.fill_between(curves["suction_kpa"],curves["theta_2.5"],curves["theta_97.5"],alpha=.25)
    ax.plot(curves["suction_kpa"],curves["theta_50"],label="Bootstrap median"); ax.scatter(suction,theta,s=18,label="TEROS data")
    ax.set_xscale("log"); ax.set(xlabel="Matric suction (kPa)",ylabel="Volumetric water content"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"SWCC_band.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5)); ax.fill_between(curves["suction_kpa"],curves["Ku_2.5"],curves["Ku_97.5"],alpha=.25)
    ax.plot(curves["suction_kpa"],curves["Ku_50"],label="Bootstrap median")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set(xlabel="Matric suction (kPa)",ylabel="Unsaturated hydraulic conductivity"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"Ku_band.png",dpi=300); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description="TEROS restricted-VG residual block bootstrap")
    ap.add_argument("--input",default="teros_swcc.csv")
    ap.add_argument("--output",default="teros_block_bootstrap_output")
    ap.add_argument("--bootstrap",type=int,default=5000)
    ap.add_argument("--method",choices=["stationary","circular","moving"],default="stationary")
    ap.add_argument("--block-length",type=int,default=None)
    ap.add_argument("--no-iid-comparison",action="store_true")
    ap.add_argument("--make-template",action="store_true")
    args=ap.parse_args()

    input_path=Path(args.input); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    if args.make_template:
        make_template(input_path); print(f"Template written to {input_path}"); return
    if not input_path.exists():
        make_template(Path("teros_swcc_input_data3.csv"))
        raise FileNotFoundError(f"{input_path} not found. A template was created as teros_swcc_template.csv")

    rng=np.random.default_rng(RANDOM_SEED)
    df=load_teros_data(input_path)
    time=df["time_hours"].to_numpy(float); suction=df["suction_kpa"].to_numpy(float)
    theta=df["theta"].to_numpy(float); sigma=df["sigma_theta"].to_numpy(float)

    original=fit_restricted_vg(suction,theta,sigma,n_starts=N_STARTS_ORIGINAL,rng=rng)
    std_resid=original["residuals"]/sigma
    diag=residual_diagnostics(std_resid,time)
    adjusted=prepare_adjusted_residuals(original["residuals"],sigma,p=2)
    L=choose_block_length(adjusted,args.method,args.block_length)
    block_duration=L*diag["median_sampling_hours"] if np.isfinite(diag["median_sampling_hours"]) else np.nan

    boot,runinfo=run_bootstrap(suction,original["predicted"],adjusted,sigma,original["z"],args.method,L,args.bootstrap,rng)
    summary=parameter_summary(boot,original)
    cov_free=boot[["alpha","n"]].cov(); corr_free=boot[["alpha","n"]].corr()
    cov_exp=boot[["alpha","n","m"]].cov(); corr_exp=boot[["alpha","n","m"]].corr()

    psi_grid=np.logspace(np.log10(suction[suction>0].min()),np.log10(suction.max()),N_GRID)
    curves=propagate_curves(boot,psi_grid)

    df.to_csv(outdir/"input_sorted.csv",index=False)
    diag["acf"].to_csv(outdir/"residual_ACF.csv",index=False); diag["ljung_box"].to_csv(outdir/"ljung_box.csv")
    boot.to_csv(outdir/"block_bootstrap_parameters.csv",index=False); summary.to_csv(outdir/"parameter_summary.csv",index=False)
    cov_free.to_csv(outdir/"covariance_alpha_n.csv"); corr_free.to_csv(outdir/"correlation_alpha_n.csv")
    cov_exp.to_csv(outdir/"covariance_alpha_n_m.csv"); corr_exp.to_csv(outdir/"correlation_alpha_n_m.csv")
    curves.to_csv(outdir/"SWCC_Kr_Ku_bands.csv",index=False)

    iid_comp=None
    if not args.no_iid_comparison:
        iid,_=run_bootstrap(suction,original["predicted"],adjusted,sigma,original["z"],"iid",1,args.bootstrap,rng)
        siid=parameter_summary(iid,original)
        iid_comp=siid[["parameter","bootstrap_SE","CI_width"]].merge(summary[["parameter","bootstrap_SE","CI_width"]],on="parameter",suffixes=("_iid","_block"))
        iid_comp["SE_ratio_block_over_iid"]=iid_comp["bootstrap_SE_block"]/iid_comp["bootstrap_SE_iid"]
        iid_comp["CI_width_ratio_block_over_iid"]=iid_comp["CI_width_block"]/iid_comp["CI_width_iid"]
        iid_comp.to_csv(outdir/"IID_vs_block.csv",index=False)

    with pd.ExcelWriter(outdir/"TEROS_block_bootstrap_results.xlsx",engine="openpyxl") as xw:
        df.to_excel(xw,sheet_name="Input_sorted",index=False); summary.to_excel(xw,sheet_name="Parameter_summary",index=False)
        diag["acf"].to_excel(xw,sheet_name="Residual_ACF",index=False); diag["ljung_box"].to_excel(xw,sheet_name="Ljung_Box")
        cov_free.to_excel(xw,sheet_name="Cov_alpha_n"); corr_free.to_excel(xw,sheet_name="Corr_alpha_n")
        cov_exp.to_excel(xw,sheet_name="Cov_alpha_n_m"); corr_exp.to_excel(xw,sheet_name="Corr_alpha_n_m")
        curves.to_excel(xw,sheet_name="SWCC_Kr_Ku",index=False)
        if iid_comp is not None: iid_comp.to_excel(xw,sheet_name="IID_vs_block",index=False)

    make_plots(outdir,time,std_resid,diag["acf"],suction,theta,boot,curves)

    report={"N":len(df),"free_parameters":["alpha","n"],"derived_parameter":"m=1-1/n","residual_df":len(df)-2,
            "alpha":original["alpha"],"n":original["n"],"m":original["m"],"R2":original["r2"],
            "bootstrap_method":args.method,"successful_bootstraps":args.bootstrap,"block_length_observations":L,
            "block_duration_hours":block_duration,"failed_attempts":runinfo["failed"],"failure_rate":runinfo["failure_rate"],
            "Durbin_Watson":diag["durbin_watson"],"effective_N_approx":diag["effective_n_approx"],
            "median_sampling_hours":diag["median_sampling_hours"],"sampling_interval_CV":diag["sampling_interval_cv"],
            "interval_method":"95% pointwise percentile bootstrap","theta_r_fixed":THETA_R,"theta_s_fixed":THETA_S,"Ks_fixed":KS,"Mualem_l_fixed":PORE_CONNECTIVITY_L}
    with open(outdir/"analysis_summary.json","w") as f: json.dump(report,f,indent=2)

    print("\nTEROS block-bootstrap analysis completed")
    print(f"N={len(df)}; alpha={original['alpha']:.6g}; n={original['n']:.6g}; m={original['m']:.6g}; R2={original['r2']:.6g}")
    print(f"Method={args.method}; block length={L} observations (~{block_duration:.6g} h)")
    print(f"Successful fits={args.bootstrap}; failed attempts={runinfo['failed']}")
    print(f"Outputs: {outdir.resolve()}")

if __name__ == "__main__":
    main()
