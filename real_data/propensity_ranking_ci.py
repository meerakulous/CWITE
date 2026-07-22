import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def load_split(data_dir, split, name):
    return joblib.load(Path(data_dir) / f"real_labor_{name}_{split}.joblib")


def fit_propensity(data_dir):
    x_train = load_split(data_dir, "train", "X")
    e_train = load_split(data_dir, "train", "binary_y")
    x_val = load_split(data_dir, "val", "X")
    e_val = load_split(data_dir, "val", "binary_y")
    x_test = load_split(data_dir, "test", "X")
    e_test = load_split(data_dir, "test", "binary_y")

    cs = [1, 1e-2, 1e-4, 1e-6]
    val_aurocs = []
    for c in cs:
        clf = LogisticRegression(random_state=0, C=c, max_iter=1000).fit(x_train, e_train)
        val_aurocs.append(roc_auc_score(e_val, clf.predict_proba(x_val)[:, 1]))

    best_c = cs[int(np.argmax(val_aurocs))]
    clf = LogisticRegression(random_state=0, C=best_c, max_iter=1000).fit(x_train, e_train)
    return e_test.astype(int), clf.predict_proba(x_test)[:, 1], best_c, max(val_aurocs)


def bootstrap_scalar(y, p, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    point = fn(y, p)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        val = fn(y[idx], p[idx])
        if np.isfinite(val):
            vals.append(float(val))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(center - half), float(center + half)


def bin_rows(y, p, bins):
    rows = []
    for lo, hi in bins:
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        k = int(y[mask].sum())
        observed = k / n if n else np.nan
        lo_ci, hi_ci = wilson_ci(k, n)
        rows.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, 1.0]",
                "n": n,
                "mean_predicted": float(np.mean(p[mask])) if n else np.nan,
                "observed_uncensoring_rate": float(observed),
                "observed_ci_low": lo_ci,
                "observed_ci_high": hi_ci,
            }
        )
    return rows


def fmt_pct(x):
    return "NA" if not np.isfinite(x) else f"{100 * x:.1f}"


def main():
    parser = argparse.ArgumentParser(description="Propensity-model AUROC, calibration, and ranking CIs.")
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/propensity_ranking_ci")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, p, best_c, val_auroc = fit_propensity(args.data_dir)

    auroc = bootstrap_scalar(y, p, roc_auc_score, args.n_boot, args.seed)
    brier = bootstrap_scalar(y, p, brier_score_loss, args.n_boot, args.seed)
    spearman = bootstrap_scalar(y, p, lambda yy, pp: spearmanr(pp, yy).correlation, args.n_boot, args.seed)
    bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
    rows = bin_rows(y, p, bins)

    summary_rows = [
        {"metric": "best_C", "estimate": best_c, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "val_AUROC_for_C_selection", "estimate": val_auroc, "ci_low": np.nan, "ci_high": np.nan},
        {"metric": "test_AUROC", "estimate": auroc[0], "ci_low": auroc[1], "ci_high": auroc[2]},
        {"metric": "test_Brier", "estimate": brier[0], "ci_low": brier[1], "ci_high": brier[2]},
        {"metric": "test_Spearman_propensity_vs_uncensored", "estimate": spearman[0], "ci_low": spearman[1], "ci_high": spearman[2]},
    ]
    with (out_dir / "propensity_summary_ci.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "estimate", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "propensity_bin_ranking_ci.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin", "n", "mean_predicted", "observed_uncensoring_rate", "observed_ci_low", "observed_ci_high"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "\\begin{table}",
        "\\caption{Propensity-model calibration by predicted propensity bin on the test set.}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Predicted propensity bin & n & Mean predicted & Observed uncensoring rate \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['bin']} & {r['n']} & {r['mean_predicted']:.3f} & "
            f"{fmt_pct(r['observed_uncensoring_rate'])}\\% "
            f"[{fmt_pct(r['observed_ci_low'])}, {fmt_pct(r['observed_ci_high'])}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (out_dir / "propensity_bin_ranking_table.tex").write_text("\n".join(lines) + "\n")

    print(f"Best C: {best_c}")
    print(f"Test AUROC: {auroc[0]:.4f} [{auroc[1]:.4f}, {auroc[2]:.4f}]")
    print(f"Test Brier: {brier[0]:.4f} [{brier[1]:.4f}, {brier[2]:.4f}]")
    print(f"Test Spearman(propensity, uncensored): {spearman[0]:.4f} [{spearman[1]:.4f}, {spearman[2]:.4f}]")
    print("\nBin ranking table:")
    for r in rows:
        print(
            f"{r['bin']}: n={r['n']}, mean_pred={r['mean_predicted']:.3f}, "
            f"observed={fmt_pct(r['observed_uncensoring_rate'])}% "
            f"[{fmt_pct(r['observed_ci_low'])}, {fmt_pct(r['observed_ci_high'])}]"
        )
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
