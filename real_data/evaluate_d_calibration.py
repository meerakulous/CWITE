import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import chisquare


DEFAULT_MODELS = [
    ("IPCW", "/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_dist.joblib"),
    ("DeepHit", "/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_dist.joblib"),
    ("Powell", "/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_dist.joblib"),
    ("CWITE", "/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_dist.joblib"),
]


def load_split(data_dir, split, name):
    return joblib.load(Path(data_dir) / f"real_labor_{name}_{split}.joblib")


def fit_propensity(data_dir):
    x_train = load_split(data_dir, "train", "X")
    e_train = load_split(data_dir, "train", "binary_y")
    x_val = load_split(data_dir, "val", "X")
    e_val = load_split(data_dir, "val", "binary_y")
    x_test = load_split(data_dir, "test", "X")
    cs = [1, 1e-2, 1e-4, 1e-6]
    val_aurocs = []
    for c in cs:
        clf = LogisticRegression(random_state=0, C=c, max_iter=1000).fit(x_train, e_train)
        val_aurocs.append(roc_auc_score(e_val, clf.predict_proba(x_val)[:, 1]))
    best_c = cs[int(np.argmax(val_aurocs))]
    clf = LogisticRegression(random_state=0, C=best_c, max_iter=1000).fit(x_train, e_train)
    return clf.predict_proba(x_test)[:, 1], best_c, max(val_aurocs)


def parse_named_paths(items, defaults):
    if not items:
        return list(defaults)
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --model value {item!r}; expected Name=/path/to/file.joblib")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def as_distribution(arr, allow_point_mass=False):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if not allow_point_mass:
            raise ValueError(
                "file contains 1D point predictions, not a predictive distribution. "
                "D-calibration needs a 2D PMF/CDF/survival array. Use --allow-point-mass "
                "only for a degenerate diagnostic, not reportable D-calibration."
            )
        max_t = int(np.nanmax(arr))
        pmf = np.zeros((len(arr), max_t + 1), dtype=float)
        idx = np.clip(np.rint(arr).astype(int), 0, max_t)
        pmf[np.arange(len(arr)), idx] = 1.0
        return pmf, "point_mass"

    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape={arr.shape}")

    row_sums = np.nansum(arr, axis=1)
    nonnegative = np.nanmin(arr) >= -1e-8
    if nonnegative and np.allclose(row_sums, 1.0, atol=1e-3):
        return np.clip(arr, 0.0, 1.0), "pmf"

    diffs = np.diff(arr, axis=1)
    if nonnegative and np.nanmin(diffs) >= -1e-6:
        cdf = np.clip(arr, 0.0, 1.0)
        pmf = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
        return np.clip(pmf, 0.0, 1.0), "cdf"

    if nonnegative and np.nanmax(diffs) <= 1e-6:
        survival = np.clip(arr, 0.0, 1.0)
        cdf = 1.0 - survival
        pmf = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
        return np.clip(pmf, 0.0, 1.0), "survival"

    raise ValueError(
        "could not infer distribution type. Expected PMF rows summing to 1, "
        "monotone increasing CDF, or monotone decreasing survival probabilities."
    )


def predicted_cdf_at_time(pmf, times):
    cdf = np.cumsum(pmf, axis=1)
    idx = np.clip(np.rint(times).astype(int), 0, cdf.shape[1] - 1)
    return cdf[np.arange(len(times)), idx]


def add_interval_mass(counts, low, high, edges):
    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(high, 0.0, 1.0))
    if high <= low:
        counts[min(len(counts) - 1, np.searchsorted(edges, low, side="right") - 1)] += 1.0
        return
    for b in range(len(counts)):
        overlap = max(0.0, min(high, edges[b + 1]) - max(low, edges[b]))
        if overlap > 0:
            counts[b] += overlap / (high - low)


def d_calibration(y, e, pmf, eval_mask, n_bins):
    cdf_values = predicted_cdf_at_time(pmf[eval_mask], y[eval_mask])
    cdf_values = np.clip(cdf_values, 0.0, np.nextafter(1.0, 0.0))
    e_eval = e[eval_mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)

    uncensored_u = cdf_values[e_eval == 1]
    uncensored_counts, _ = np.histogram(uncensored_u, bins=edges)
    counts += uncensored_counts.astype(float)

    for u in cdf_values[e_eval == 0]:
        add_interval_mass(counts, u, 1.0, edges)

    n_eval = int(np.sum(eval_mask))
    n_uncensored = int(np.sum(e_eval == 1))
    n_censored = int(np.sum(e_eval == 0))
    expected = np.repeat(n_eval / n_bins, n_bins)
    stat, p_value = chisquare(counts, expected)
    max_abs_bin_error = float(np.max(np.abs(counts / n_eval - 1.0 / n_bins))) if n_eval else np.nan
    return {
        "n": n_eval,
        "n_uncensored": n_uncensored,
        "n_censored": n_censored,
        "chi_square": float(stat),
        "p_value": float(p_value),
        "max_abs_bin_error": max_abs_bin_error,
        "counts": counts,
        "edges": edges,
    }


def make_latex(rows, n_bins):
    lines = [
        "\\begin{table}",
        "\\caption{D-calibration on the real-data test set. The test is computed within each evaluation group by binning predicted CDF values into "
        f"{n_bins} equal-width bins on $[0,1]$; censored individuals contribute interval mass over $[\\hat F(C_i),1]$.}}",
        "\\label{tab:realdata_d_calibration}",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Group & Model & Distribution input & $\\chi^2$ & $p$-value \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {row['model']} & {row['distribution_type']} & "
            f"{row['chi_square']:.2f} & {row['p_value']:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute D-calibration for survival models that output full predictive distributions. "
            "For uncensored individuals, predicted CDF values at observed TTE should be uniform."
        )
    )
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/cwite_realdata_final/d_calibration")
    parser.add_argument("--model", action="append", default=[], help="Model as Name=/path/to/dist.joblib")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--allow-point-mass", action="store_true", help="Allow 1D point predictions as degenerate distributions; diagnostic only.")
    parser.add_argument("--no-defaults", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(load_split(args.data_dir, "test", "y"), dtype=float)
    e = np.asarray(load_split(args.data_dir, "test", "binary_y"), dtype=int)
    propensity, best_c, val_auroc = fit_propensity(args.data_dir)
    groups = {
        "All test": np.ones(len(y), dtype=bool),
        f"Propensity [0.00, {args.low_max:.2f})": propensity < args.low_max,
        f"Propensity [{args.high_min:.2f}, 1.00]": propensity >= args.high_min,
    }
    model_paths = parse_named_paths(args.model, [] if args.no_defaults else DEFAULT_MODELS)
    print(f"Propensity model: best C={best_c}, val AUROC={val_auroc:.4f}", flush=True)

    rows = []
    bin_rows = []
    for model, path in model_paths:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: missing distribution file for {model}: {p}", flush=True)
            continue
        try:
            pmf, dist_type = as_distribution(joblib.load(p), allow_point_mass=args.allow_point_mass)
            if len(pmf) != len(y):
                raise ValueError(f"{model} has {len(pmf)} rows but y_test has {len(y)}")
        except Exception as exc:
            print(f"WARNING: skipping {model}: {exc}", flush=True)
            continue

        for group, mask in groups.items():
            if not np.any(mask):
                continue
            result = d_calibration(y, e, pmf, mask, args.n_bins)
            row = {
                "group": group,
                "model": model,
                "path": str(p),
                "distribution_type": dist_type,
                "n": result["n"],
                "n_uncensored": result["n_uncensored"],
                "n_censored": result["n_censored"],
                "n_bins": args.n_bins,
                "chi_square": result["chi_square"],
                "p_value": result["p_value"],
                "max_abs_bin_error": result["max_abs_bin_error"],
            }
            rows.append(row)
            print(
                f"{group} | {model}: type={dist_type}, n={row['n']}, "
                f"uncensored={row['n_uncensored']}, censored={row['n_censored']}, "
                f"chi2={row['chi_square']:.3f}, p={row['p_value']:.4f}, "
                f"max_abs_bin_error={row['max_abs_bin_error']:.4f}",
                flush=True,
            )
            for b, count in enumerate(result["counts"]):
                bin_rows.append(
                    {
                        "group": group,
                        "model": model,
                        "bin": b + 1,
                        "bin_low": result["edges"][b],
                        "bin_high": result["edges"][b + 1],
                        "count": float(count),
                        "expected": result["n"] / args.n_bins,
                    }
                )

    if not rows:
        raise FileNotFoundError(
            "No D-calibration results were computed. Pass --model Name=/path/to/2d_distribution.joblib "
            "files, or rerun training with saved test PMFs/CDFs."
        )

    with (out_dir / "d_calibration_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "d_calibration_bins.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    tex = make_latex(rows, args.n_bins)
    (out_dir / "d_calibration_table.tex").write_text(tex + "\n")
    print(f"Saved CSV/LaTeX outputs to {out_dir}", flush=True)
    print("\n" + tex, flush=True)


if __name__ == "__main__":
    main()
