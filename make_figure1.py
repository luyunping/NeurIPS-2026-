#!/usr/bin/env python3
"""
make_figure1.py
===============
Render a Figure-1-style two-panel plot from the per-repetition records
produced by figure1_reproduction.py (file ending in _records.csv).

Left panel : per method (oracle / proposed / baseline) at the largest N --
             median point estimate and mean 95% CI width, against the true
             value (dashed red line).
Right panel: sampling distribution of the proposed estimator across sample
             sizes N (boxplots over repetitions), showing sqrt{N}-type
             contraction toward the truth.

Usage:
    python3 make_figure1.py --records fig1_records.csv --out figure1.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

METHODS = ["oracle", "proposed", "baseline"]
COLORS = {"oracle": "#4C9F70", "proposed": "#2E6FDB", "baseline": "#C0563B"}


def load(records_path):
    data = defaultdict(list)   # (N, method) -> list of (est, ci_half, truth)
    with open(records_path) as fh:
        import csv
        for row in csv.DictReader(fh):
            key = (int(row["N"]), row["method"])
            data[key].append((float(row["est"]), float(row["ci_half"]),
                              float(row["theta_true"])))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default="figure1.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    data = load(args.records)
    Ns = sorted({k[0] for k in data})
    N_star = max(Ns)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # ---- left: estimation error and CI widths at N_star ----
    ax = axes[0]
    ax.axhline(0.0, color="red", ls="--", lw=1.2, label="True value (0)")
    y_lo, y_hi = 0.0, 0.0
    stats = {}
    for meth in METHODS:
        rec = data[(N_star, meth)]
        err = np.array([t[0] - t[2] for t in rec])     # estimate - truth
        half = np.array([t[1] for t in rec])
        med = float(np.median(err))
        width = float(2 * half.mean())
        stats[meth] = (med, width)
        y_lo = min(y_lo, med - width / 2)
        y_hi = max(y_hi, med + width / 2)
    pad = 0.35 * (y_hi - y_lo + 1e-9)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    for i, meth in enumerate(METHODS):
        med, width = stats[meth]
        ax.errorbar(i, med, yerr=width / 2, fmt="o", color=COLORS[meth],
                    ecolor=COLORS[meth], elinewidth=6, capsize=0, alpha=0.85)
        ax.annotate(f"[{med - width/2:.3f}, {med + width/2:.3f}]",
                    (i, med + width / 2), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
        ax.annotate(f"Width: {width:.3f}", (i, med - width / 2),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8)
        print(f"N={N_star} {meth}: median error={med:+.3f} "
              f"CI width={width:.3f}")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(["Oracle", "Proposed (Ours)", "Baseline"])
    ax.set_ylabel("Estimation error (estimate − truth)")
    ax.set_title(f"Point estimates and 95% CIs (N={N_star}, "
                 f"{len(rec)} reps)")
    ax.legend(loc="best", fontsize=8)

    # ---- right: convergence of the proposed estimator across N ----
    ax = axes[1]
    series, labels = [], []
    for N in Ns:
        rec = data[(N, "proposed")]
        est = np.array([t[0] for t in rec])
        tru = np.array([t[2] for t in rec])
        series.append(est - tru)          # estimation error, centered at 0
        labels.append(str(N))
    bp = ax.boxplot(series, tick_labels=labels, showmeans=True,
                    medianprops=dict(color="red", lw=1.5))
    ax.axhline(0.0, color="red", ls="--", lw=1.2)
    for i, s in enumerate(series):
        q = np.percentile(s, [2.5, 97.5])
        ax.text(i + 1, ax.get_ylim()[1] * 0.92,
                f"{s.mean():+.3f}±{s.std(ddof=1):.3f}", ha="center", fontsize=8)
        print(f"N={labels[i]}: mean={s.mean():+.3f} sd={s.std(ddof=1):.3f} "
              f"empirical 95% width={q[1]-q[0]:.3f}")
    ax.set_xlabel("Total sample size N")
    ax.set_ylabel("Estimation error (estimate − truth)")
    ax.set_title("Convergence of the localized DR-Lasso estimator")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
