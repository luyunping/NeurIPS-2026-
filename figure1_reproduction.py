#!/usr/bin/env python3
"""
figure1_reproduction.py
=======================
Re-implementation of the synthetic experiment pipeline underlying Figure 1 and
Appendix B.1 of "Individualized Causal Effects under Network Interference with
Combinatorial Treatments".

Pipeline (Algorithm 2 of the paper):
  1. Rooted network configurations with the truncated rooted-graph distance d_R
     (R = 1, star approximation of the ego network; root's OWN slate is excluded
     from the marks so that localization preserves own-slate variation needed
     for local overlap, Assumption 3);
  2. Kernel / kNN localization weights and the Kish effective sample size n_eff;
  3. Cross-fitted doubly-robust residualization of Y and Z(T) (Eq. 6);
  4. Localized weighted Lasso over the Walsh-Hadamard dictionary (Eq. 7);
  5. Debiased inference for a user-specified own-treatment contrast (Eqs. 9-10),
     with the variance estimator of Theorem 4.

Comparators (Appendix B):
  * Oracle   : knows the true nuisance functions and the true Walsh support;
  * Baseline : graph-agnostic DR learner + simple network averaging.

Fidelity notes (documented simplifications):
  * Configurations use the radius-1 star approximation of Appendix A (edges
    among neighbors are ignored); the mismatch fraction under the optimal
    matching is computed EXACTLY via multiset-count intersection, which equals
    the minimum over root-preserving bijections for unit mismatch costs.
  * m(g,x) = E[Z(T)|G,X] is estimated by fold-wise means, which is exact under
    the uniform random assignment design of Appendix B.
  * The Walsh dictionary is truncated at interaction order 3 by default
    (hierarchical truncation is explicitly allowed by the paper); pass
    "--order full" for the complete 2^p dictionary.
  * lambda is chosen by weighted cross-validation (LassoCV); the theoretical
    choice lambda ~ sigma*sqrt(log d / n_eff) is used for the debiasing
    tolerance eta (Eq. 9).

IMPORTANT: report whatever numbers this code actually produces. Do not tune
the DGP to hit pre-specified values.
"""

import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------
# Walsh-Hadamard dictionary
# --------------------------------------------------------------------------

def subsets_upto(p, order):
    """Bitmask ids of subsets S of [p] with |S| <= order (None -> all 2^p)."""
    if order is None:
        return np.arange(2 ** p, dtype=np.int64)
    out = [s for s in range(2 ** p) if bin(s).count("1") <= order]
    return np.array(out, dtype=np.int64)


def all_slates(p):
    """All 2^p slates as a (2^p, p) matrix with entries in {-1, +1}."""
    ids = np.arange(2 ** p)
    return 1 - 2 * ((ids[:, None] >> np.arange(p)) & 1)


def slate_ids(T):
    """Map each row of T (N x p, entries +/-1) to its integer id in [0, 2^p)."""
    return (T < 0).dot(1 << np.arange(T.shape[1]))


def walsh_matrix(T, subsets):
    """Z_S(t) = prod_{l in S} t_l, for rows t of T and bitmask list `subsets`."""
    B = (T < 0).astype(np.int64)                          # N x p, bit=1 where t=-1
    SB = (subsets[:, None] >> np.arange(T.shape[1])) & 1  # d x p
    return 1 - 2 * ((B @ SB.T) % 2)                       # N x d, entries +/-1


# --------------------------------------------------------------------------
# Rooted configurations and the truncated rooted-graph distance (R = 1)
# --------------------------------------------------------------------------

def config_counts(A, T, n_slates):
    """Neighbor-slate count vector for each unit (marks = neighbors' slates).

    The root's own slate is deliberately EXCLUDED from the configuration:
    the estimand theta^T varies the own slate holding the interference
    environment fixed, so localization must be over the environment only.
    """
    ids = slate_ids(T)
    N = A.shape[0]
    C = np.zeros((N, n_slates))
    deg = A.sum(1).astype(int)
    for i in range(N):
        nb = np.flatnonzero(A[i])
        if nb.size:
            C[i] = np.bincount(ids[nb], minlength=n_slates)
    return C, deg


def rooted_distances(C, deg, target):
    """d_R(g_j, g_target) with R = 1 (Appendix A, star approximation).

    Delta_0 = 0 (root mark excluded). For equal degrees k, the mark-mismatch
    fraction under the optimal matching is sum_v min(c_v, c'_v) matches, i.e.
    Delta_1 = ||c - c'||_1 / (2k); if degrees differ, Delta_1 = 1.
    d_R = 2^{-1}*Delta_0 + 2^{-2}*Delta_1 = Delta_1 / 4.
    """
    d1 = cdist(C, C[[target]], metric="cityblock").ravel()
    k = deg[target]
    if k == 0:
        delta1 = np.where(deg == 0, 0.0, 1.0)
    else:
        delta1 = np.where(deg == k, d1 / (2.0 * k), 1.0)
    return 0.25 * delta1


# --------------------------------------------------------------------------
# Data-generating process (Appendix B / B.1)
# --------------------------------------------------------------------------

def dgp(N, p, avg_deg, sigma, c_het, seed, random_coefs=False, interf=1.0):
    """Generate one snapshot (Y, T, X, A) plus oracle quantities.

    Y_i = <alpha(g_i), Z(t_i)> + interference(mean neighbor slate) + X_i'gamma + eps
    with a sparse active set and one configuration-heterogeneous main effect
    a_{0}(g) = c_het * (frac_nb(g) - frac_nb(g_target)), centered so that the
    target contrast (flip coordinate 0) has true value 0, as in the paper.

    If random_coefs=True, the ground-truth coefficients themselves are drawn
    from the RNG on every call, so that literally nothing in the simulation
    is hand-set; the estimates are always computed from the realized data.
    """
    rng = np.random.default_rng(seed)

    # Erdos-Renyi network, average degree avg_deg
    U = rng.random((N, N))
    A = np.triu((U < avg_deg / (N - 1)).astype(float), 1)
    A = A + A.T

    X = rng.normal(size=(N, p))
    T = rng.choice([-1.0, 1.0], size=(N, p))

    C, deg = config_counts(A, T, 2 ** p)
    V = all_slates(p)                                     # slate value per id
    with np.errstate(invalid="ignore", divide="ignore"):
        nb_mean = np.where(deg[:, None] > 0, (C @ V) / np.maximum(deg, 1)[:, None], 0.0)
    frac_nb = ((nb_mean + 1.0) / 2.0).mean(1)             # mean treated fraction

    # Target unit: median interference environment
    target = int(np.argmin(np.abs(frac_nb - np.median(frac_nb))))

    # Own-slate coefficients (sparse; coordinate 0 heterogeneous in g)
    if random_coefs:
        signs = rng.choice([-1.0, 1.0], size=3)
        a1, a2, a34 = signs * rng.uniform(0.4, 1.0, size=3)
        b1, b2 = rng.uniform(0.3, 0.8, size=2)
        gamma = rng.normal(0.0, 0.3, size=p)
    else:
        a1, a2, a34 = 0.8, -0.6, 0.5                      # coords {1}, {2}, {3,4}
        b1, b2 = 0.7, 0.4
        gamma = np.full(p, 0.3)
    a0_g = c_het * (frac_nb - frac_nb[target])            # zero at the target
    own = a0_g * T[:, 0] + a1 * T[:, 1] + a2 * T[:, 2]
    support = [1 << 0, 1 << 1, 1 << 2]
    alpha_vals = [a0_g[target], a1, a2]
    if p >= 5:
        own = own + a34 * T[:, 3] * T[:, 4]
        support.append((1 << 3) | (1 << 4))
        alpha_vals.append(a34)

    # First-order neighbor interference through the mean neighbor slate
    if p >= 4:
        inter = interf * (b1 * nb_mean[:, 1] + b2 * nb_mean[:, 2] * nb_mean[:, 3])
    else:
        inter = interf * (b1 * nb_mean[:, 1] + b2 * nb_mean[:, 1] * nb_mean[:, 2])

    mu_x = X @ gamma
    eps = rng.normal(0.0, sigma, N)
    Y = own + inter + mu_x + eps

    # User-specified own-treatment contrast: flip coordinate 0, all else at -1
    t0 = -np.ones(p)
    t1 = t0.copy(); t1[0] = 1.0

    # Oracle quantities
    mu_true = inter + mu_x                                # E[Y | G, X]
    support = np.array(support)
    alpha_target = np.array(alpha_vals)
    theta_true = float(2.0 * a0_g[target])                # = 0 by construction

    return dict(Y=Y, T=T, X=X, A=A, C=C, deg=deg, nb_mean=nb_mean,
                target=target, t0=t0, t1=t1, mu_true=mu_true,
                support=support, alpha_target=alpha_target,
                theta_true=theta_true)


# --------------------------------------------------------------------------
# Localization weights and nuisances
# --------------------------------------------------------------------------

def kernel_weights(d, bandwidth):
    """Epanechnikov kernel with FIXED bandwidth b_G on the d_R scale (Eq. 1).

    NOTE: under the ER + random-slate design, d_R equals its maximum for the
    large majority of configuration pairs, so kNN/adaptive bandwidths collapse
    the effective sample size to a handful of near-duplicate configurations.
    The paper's reported b_G = 2 yields mild, non-degenerate localization
    (n_eff ~ N); smaller values give stronger localization and should be
    reported as a sensitivity analysis, with n_eff quoted alongside.
    """
    u = d / max(bandwidth, 1e-8)
    w = np.where(u < 1.0, 1.0 - u ** 2, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def crossfit_mu(F, Y, n_folds, seed):
    """Cross-fitted mu(g,x) = E[Y | G, X] via gradient boosting on features
    [X, degree, mean neighbor slate] (a sufficient featurization of g here)."""
    mu = np.zeros_like(Y)
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(F):
        m = HistGradientBoostingRegressor(max_iter=200)
        m.fit(F[tr], Y[tr])
        mu[te] = m.predict(F[te])
    return mu


def crossfit_m(Z, n_folds, seed):
    """Cross-fitted m(g,x) = E[Z(T) | G, X] by fold means (exact under the
    uniform random assignment design of Appendix B)."""
    m = np.zeros_like(Z)
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(Z):
        m[te] = Z[tr].mean(0)
    return m


# --------------------------------------------------------------------------
# Localized weighted Lasso and debiased inference (Eqs. 7, 9, 10, Thm 4)
# --------------------------------------------------------------------------

def fit_lasso(Zr, Yr, w, seed, c_lambda=1.5):
    """Weighted Lasso (Eq. 7) with the paper's theoretical penalty
    lambda = c * sigma_hat * sqrt(log d / n_eff) (sigma_hat from a CV pilot).
    Cross-validated lambda proved unstable at small n_eff (occasionally
    selecting alpha ~ 0, which inflates dense contrast directions v)."""
    n, d = Zr.shape
    n_eff = 1.0 / np.sum(w ** 2)
    sw = np.sqrt(w)
    Zw, Yw = Zr * sw[:, None], Yr * sw

    cv = LassoCV(alphas=15, cv=3, fit_intercept=False, max_iter=5000,
                 tol=1e-3, random_state=seed)
    cv.fit(Zw, Yw)
    resid = Yw - Zw @ cv.coef_
    sigma_hat = float(np.sqrt(np.sum(resid ** 2)))  # sum w_j r_j^2 ~ sigma^2

    lam = c_lambda * sigma_hat * np.sqrt(np.log(d) / n_eff)
    alpha_sk = lam / (2.0 * n)   # paper: min sum w r^2 + lam||b||_1
    m = Lasso(alpha=alpha_sk, fit_intercept=False, max_iter=20000, tol=1e-4)
    m.fit(Zw, Yw)
    return m.coef_


def debias_contrast(Zr, Yr, w, alpha, v, eta_mult=0.5):
    """Debiased own-treatment contrast (Eqs. 9-10) + Thm-4 variance.

    Returns (theta_debiased, ci_half_width). n enters as n_eff = 1/sum(w^2).
    """
    d = Zr.shape[1]
    n_eff = 1.0 / np.sum(w ** 2)
    Sig = (Zr * w[:, None]).T @ Zr
    eta = eta_mult * np.sqrt(np.log(d) / n_eff)

    # Eq. (9) as an LP:  min ||g||_1  s.t. ||Sig g - v||_inf <= eta
    A_ub = np.block([[Sig, np.zeros((d, d))],
                     [-Sig, np.zeros((d, d))],
                     [np.eye(d), -np.eye(d)],
                     [-np.eye(d), -np.eye(d)]])
    b_ub = np.concatenate([v + eta, -v + eta, np.zeros(2 * d)])
    c = np.concatenate([np.zeros(d), np.ones(d)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * d + [(0, None)] * d, method="highs")
    gamma = res.x[:d] if res.success else np.linalg.lstsq(
        Sig + 1e-6 * np.eye(d), v, rcond=None)[0]

    eps = Yr - Zr @ alpha
    theta = float(v @ alpha + gamma @ ((Zr * (w * eps)[:, None]).sum(0)))
    infl = w * (Zr @ gamma) * eps
    ci_half = 1.96 * float(np.sqrt(np.sum(infl ** 2)))
    return theta, ci_half


def proposed_estimator(data, subsets, v, bandwidth, n_folds, seed):
    """Full Algorithm 2: localize + DR residualize + Lasso + debias."""
    Y, T, X = data["Y"], data["T"], data["X"]
    F = np.column_stack([X, data["deg"], data["nb_mean"]])
    Z = walsh_matrix(T, subsets)
    mu = crossfit_mu(F, Y, n_folds, seed)
    m = crossfit_m(Z, n_folds, seed)
    Yr, Zr = Y - mu, Z - m

    d = rooted_distances(data["C"], data["deg"], data["target"])
    w = kernel_weights(d, bandwidth)

    alpha = fit_lasso(Zr, Yr, w, seed)
    return debias_contrast(Zr, Yr, w, alpha, v)


def oracle_estimator(data, subsets, v, bandwidth):
    """Infeasible benchmark: true nuisances and true Walsh support known."""
    Y, T = data["Y"], data["T"]
    Z = walsh_matrix(T, subsets)
    Yr = Y - data["mu_true"]
    Zr = Z - Z.mean(0)                                  # oracle m
    d = rooted_distances(data["C"], data["deg"], data["target"])
    w = kernel_weights(d, bandwidth)

    cols = np.array([np.where(subsets == s)[0][0] for s in data["support"]
                     if s in subsets])
    Zc = Zr[:, cols]
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(Zc * sw[:, None], Yr * sw, rcond=None)[0]

    theta = float(beta @ v[cols])
    eps = Yr - Zc @ beta
    G = (Zc * w[:, None]).T @ Zc
    H = np.linalg.solve(G + 1e-10 * np.eye(len(cols)), v[cols])
    infl = w * (Zc @ H) * eps
    ci_half = 1.96 * float(np.sqrt(np.sum(infl ** 2)))
    return theta, ci_half


def baseline_estimator(data, subsets, v, n_folds, seed):
    """Graph-agnostic DR learner + simple network averaging (reconstruction)."""
    Y, T, X, A = data["Y"], data["T"], data["X"], data["A"]
    deg = np.maximum(data["deg"], 1)
    Y_adj = Y - (A @ Y) / deg                             # network averaging
    Z = walsh_matrix(T, subsets)
    mu = crossfit_mu(X, Y_adj, n_folds, seed)             # X only, no graph
    m = crossfit_m(Z, n_folds, seed)
    Yr, Zr = Y_adj - mu, Z - m

    w = np.full(len(Y), 1.0 / len(Y))                     # no localization
    alpha = fit_lasso(Zr, Yr, w, seed)
    return debias_contrast(Zr, Yr, w, alpha, v)


# --------------------------------------------------------------------------
# Monte Carlo driver
# --------------------------------------------------------------------------

METHODS = ("oracle", "proposed", "baseline")


def make_direction(data, subsets, contrast, coef_subset):
    """Contrast direction v and its true value.

    contrast='flip': v = Z(t1)-Z(t0) for the slate flip in data['t0']/data['t1']
        (the paper's headline theta^T; NOTE: v is dense on the dictionary, so
        gamma* is dense and the debiased variance scales with ||v||_2 -- this
        is the regime Theorem 4(iii) excludes unless n_eff is very large).
    contrast='coef': v = e_{S*} for a single Walsh coefficient (sparse gamma*,
        the regime where debiased inference is feasible at moderate n_eff).
    """
    if contrast == "flip":
        v = (walsh_matrix(data["t1"][None], subsets)
             - walsh_matrix(data["t0"][None], subsets)).ravel()
        theta_true = data["theta_true"]
    else:
        v = (subsets == coef_subset).astype(float)
        hit = np.where(data["support"] == coef_subset)[0]
        theta_true = float(data["alpha_target"][hit[0]]) if hit.size else 0.0
    return v, theta_true


def run_one_rep(N, p, avg_deg, sigma, c_het, subsets, bandwidth, n_folds, seed,
                random_coefs=False, contrast="flip", coef_subset=1, interf=1.0):
    data = dgp(N, p, avg_deg, sigma, c_het, seed, random_coefs=random_coefs,
               interf=interf)
    v, theta_true = make_direction(data, subsets, contrast, coef_subset)
    out = {"theta_true": theta_true}
    out["oracle"] = oracle_estimator(data, subsets, v, bandwidth)
    out["proposed"] = proposed_estimator(data, subsets, v, bandwidth, n_folds, seed)
    out["baseline"] = baseline_estimator(data, subsets, v, n_folds, seed)
    return out


def summarize(records):
    """Per method: median estimate, bias, mean CI width, coverage, MC std."""
    rows = {}
    theta_true = float(np.median([r["theta_true"] for r in records]))
    for meth in METHODS:
        est = np.array([r[meth][0] for r in records])
        half = np.array([r[meth][1] for r in records])
        truth = np.array([r["theta_true"] for r in records])
        rows[meth] = dict(
            median_est=float(np.median(est)),
            median_bias=float(np.median(est - truth)),
            mean_ci_width=float(2.0 * half.mean()),
            coverage=float(np.mean(np.abs(est - truth) <= half)),
            mc_std=float(est.std(ddof=1)),
            err_std=float((est - truth).std(ddof=1)),
        )
    return theta_true, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[50, 100, 200, 500, 1000])
    ap.add_argument("--reps", type=int, default=100,
                    help="Monte Carlo repetitions per N (paper: 100)")
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--avg-deg", type=float, default=8.0)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="noise sd; Appendix B.1 uses eps ~ N(0, 0.25)")
    ap.add_argument("--c-het", type=float, default=1.0,
                    help="strength of configuration heterogeneity")
    ap.add_argument("--order", default="3",
                    help="max Walsh interaction order, or 'full' for 2^p")
    ap.add_argument("--bandwidth", type=float, default=2.0,
                    help="Epanechnikov bandwidth b_G on the d_R scale "
                         "(paper: 2; smaller = stronger localization, report n_eff)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-coefs", action="store_true",
                    help="draw ground-truth coefficients from the RNG each rep "
                         "(recommended: nothing in the simulation is hand-set)")
    ap.add_argument("--contrast", choices=["flip", "coef"], default="coef",
                    help="'flip': slate flip (dense v, high variance); "
                         "'coef': single Walsh coefficient (sparse v)")
    ap.add_argument("--coef-subset", type=int, default=2,
                    help="bitmask of the Walsh subset for contrast='coef' "
                         "(1={coord0}, true 0; 2={coord1}, true a1; 4={coord2})")
    ap.add_argument("--interf", type=float, default=1.0,
                    help="interference strength multiplier (0 = no-interference "
                         "ablation)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="figure1_results.csv")
    args = ap.parse_args()

    order = None if args.order == "full" else int(args.order)
    subsets = subsets_upto(args.p, order)
    print(f"Walsh dictionary: p={args.p}, order={args.order}, d={len(subsets)}")

    all_rows = []
    rep_rows = []
    for N in args.Ns:
        records = []
        for r in range(args.reps):
            seed = args.seed + 1000003 * N + 997 * r
            rec = run_one_rep(N, args.p, args.avg_deg, args.sigma,
                              args.c_het, subsets, args.bandwidth,
                              args.folds, seed,
                              random_coefs=args.random_coefs,
                              contrast=args.contrast,
                              coef_subset=args.coef_subset,
                              interf=args.interf)
            records.append(rec)
            for meth in METHODS:
                rep_rows.append(dict(N=N, rep=r, method=meth,
                                     est=rec[meth][0], ci_half=rec[meth][1],
                                     theta_true=rec["theta_true"]))
            if (r + 1) % 10 == 0:
                print(f"  N={N}: {r + 1}/{args.reps} reps done", flush=True)
        theta_true, rows = summarize(records)
        for meth, s in rows.items():
            line = dict(N=N, bandwidth=args.bandwidth, theta_true=theta_true,
                        method=meth, **s)
            all_rows.append(line)
            print(f"N={N:<5d} {meth:<9s} median={s['median_est']:+.3f} "
                  f"bias={s['median_bias']:+.3f} CIw={s['mean_ci_width']:.3f} "
                  f"cov={s['coverage']:.2f} std={s['mc_std']:.3f}")

    import csv
    with open(args.out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(all_rows)
    rec_out = args.out.replace(".csv", "") + "_records.csv"
    with open(rec_out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rep_rows)
    print(f"Saved -> {args.out} and {rec_out}")


if __name__ == "__main__":
    main()
