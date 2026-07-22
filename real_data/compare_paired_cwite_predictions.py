import argparse
from pathlib import Path
from cwite_paths import real_data_dir, real_run_path, real_output_path

import joblib
import numpy as np

from generate_realdata_report_tables import fit_propensity, load_split


def paired_mae_diff(y, pred_ref, pred_cmp, idx):
    ref_err = np.abs(y[idx] - pred_ref[idx])
    cmp_err = np.abs(y[idx] - pred_cmp[idx])
    return float(np.mean(cmp_err - ref_err))


def bootstrap_ci(y, pred_ref, pred_cmp, idx, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        vals.append(paired_mae_diff(y, pred_ref, pred_cmp, sample))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_pvalue(y, pred_ref, pred_cmp, idx, n_perm, seed):
    rng = np.random.default_rng(seed)
    diffs = np.abs(y[idx] - pred_cmp[idx]) - np.abs(y[idx] - pred_ref[idx])
    obs = float(np.mean(diffs))
    if len(diffs) == 0:
        return np.nan
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs), replace=True)
        null.append(float(np.mean(signs * diffs)))
    null = np.asarray(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (len(null) + 1.0)
    return float(p)


def fmt(x):
    return "NA" if not np.isfinite(x) else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired test comparing two CWITE prediction files. "
            "Reports MAE(comparison) - MAE(reference), so positive values mean comparison is worse."
        )
    )
    parser.add_argument("--data-dir", default=real_data_dir())
    parser.add_argument(
        "--reference",
        default=real_run_path('cwite/proposed_feature_sweep_best_y_test_pred.joblib'),
        help="Reference prediction file, usually CWITE k-means.",
    )
    parser.add_argument(
        "--comparison",
        default=real_run_path('cwite_random_clusters/random_cluster_proposed_y_test_pred.joblib'),
        help="Comparison prediction file, usually CWITE random clusters.",
    )
    parser.add_argument("--reference-name", default="CWITE KMeans")
    parser.add_argument("--comparison-name", default="CWITE Random Clusters")
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    y = np.asarray(load_split(args.data_dir, "test", "y"), dtype=float)
    e = np.asarray(load_split(args.data_dir, "test", "binary_y"), dtype=int)
    propensity, best_c, val_auroc = fit_propensity(args.data_dir)

    pred_ref = np.asarray(joblib.load(args.reference), dtype=float).reshape(-1)
    pred_cmp = np.asarray(joblib.load(args.comparison), dtype=float).reshape(-1)
    if len(pred_ref) != len(y) or len(pred_cmp) != len(y):
        raise ValueError(
            f"Prediction lengths must match y: y={len(y)}, reference={len(pred_ref)}, comparison={len(pred_cmp)}"
        )

    masks = {
        "All uncensored test individuals": np.ones(len(y), dtype=bool),
        f"Propensity [0.00, {args.low_max:.2f})": propensity < args.low_max,
        f"Propensity [{args.high_min:.2f}, 1.00]": propensity >= args.high_min,
    }

    rows = []
    print(f"Reference:  {args.reference_name} ({args.reference})", flush=True)
    print(f"Comparison: {args.comparison_name} ({args.comparison})", flush=True)
    print(f"Propensity model: best C={best_c}, val AUROC={val_auroc:.4f}", flush=True)
    print("Difference is MAE(comparison) - MAE(reference); positive means comparison is worse.\n", flush=True)

    for group, mask in masks.items():
        idx = np.where(mask & (e == 1))[0]
        if len(idx) == 0:
            continue
        ref_mae = float(np.mean(np.abs(y[idx] - pred_ref[idx])))
        cmp_mae = float(np.mean(np.abs(y[idx] - pred_cmp[idx])))
        diff = cmp_mae - ref_mae
        lo, hi = bootstrap_ci(y, pred_ref, pred_cmp, idx, args.n_boot, args.seed)
        p = sign_flip_pvalue(y, pred_ref, pred_cmp, idx, args.n_perm, args.seed + 1)
        rows.append(
            {
                "group": group,
                "n_uncensored": len(idx),
                "reference_mae": ref_mae,
                "comparison_mae": cmp_mae,
                "mae_diff_comparison_minus_reference": diff,
                "ci_low": lo,
                "ci_high": hi,
                "paired_sign_flip_p": p,
            }
        )
        print(
            f"{group}: n_uncensored={len(idx)} | "
            f"{args.reference_name} MAE={ref_mae:.4f}, {args.comparison_name} MAE={cmp_mae:.4f}, "
            f"diff={diff:.4f} [{lo:.4f}, {hi:.4f}], paired p={fmt(p)}",
            flush=True,
        )

    if args.out_csv:
        import csv

        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {out_path}", flush=True)


if __name__ == "__main__":
    main()
