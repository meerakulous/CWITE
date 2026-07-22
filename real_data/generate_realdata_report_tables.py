import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from lifelines.utils import concordance_index
except ImportError:
    concordance_index = None


DEFAULT_MODELS = [
    ("IPCW", "/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_pred.joblib"),
    ("DeepHit", "/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_pred.joblib"),
    ("Powell", "/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_pred.joblib"),
    ("CWITE", "/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_pred.joblib"),
]

DEFAULT_CLUSTER_MODELS = [
    ("CWITE KMeans selected", "/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_pred.joblib"),
    ("CWITE Random Clusters", "/data4/meerak/cwite_realdata_final/cwite_random_clusters/random_cluster_proposed_y_test_pred.joblib"),
    ("CWITE Global No Clusters", "/data4/meerak/cwite_realdata_final/cwite_global_no_clusters/global_cluster_proposed_y_test_pred.joblib"),
    ("CWITE KMeans rerun", "/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_kmeans_y_test_pred.joblib"),
    ("CWITE KMeans+PCA", "/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_kmeans_pca_y_test_pred.joblib"),
    ("CWITE GMM", "/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_gmm_y_test_pred.joblib"),
    ("CWITE Spectral", "/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_spectral_y_test_pred.joblib"),
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
    scores = []
    for c in cs:
        clf = LogisticRegression(random_state=0, C=c, max_iter=1000).fit(x_train, e_train)
        scores.append(roc_auc_score(e_val, clf.predict_proba(x_val)[:, 1]))
    best_c = cs[int(np.argmax(scores))]
    clf = LogisticRegression(random_state=0, C=best_c, max_iter=1000).fit(x_train, e_train)
    return clf.predict_proba(x_test)[:, 1], best_c, max(scores)


def parse_named_paths(items, defaults):
    if not items:
        return list(defaults)
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad model argument {item!r}; expected Name=/path/to/pred.joblib")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def load_predictions(named_paths, n):
    preds = {}
    for name, path in named_paths:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: missing prediction file for {name}: {p}", flush=True)
            continue
        arr = np.asarray(joblib.load(p)).reshape(-1)
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} predictions but expected {n}")
        preds[name] = arr
    if not preds:
        raise FileNotFoundError("No prediction files were found.")
    return preds


def make_masks(propensity, low_max, high_min):
    return {
        f"Propensity [0.00, {low_max:.2f})": propensity < low_max,
        f"Propensity [{high_min:.2f}, 1.00]": propensity >= high_min,
    }


def make_clinical_masks(propensity, low_max, high_min):
    masks = {"All test individuals": np.ones(len(propensity), dtype=bool)}
    masks.update(make_masks(propensity, low_max, high_min))
    return masks


def mae_uncensored(y, e, pred):
    mask = e == 1
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(np.abs(y[mask] - pred[mask])))


def cindex(y, e, pred):
    if concordance_index is None:
        return np.nan
    if np.sum(e == 1) < 2:
        return np.nan
    try:
        return float(concordance_index(y, pred, e))
    except ValueError:
        return np.nan


def bootstrap(y, e, pred, mask, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    point = fn(y[idx_all], e[idx_all], pred[idx_all])
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        val = fn(y[idx], e[idx], pred[idx])
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return point, np.nan, np.nan, len(idx_all), int(np.sum(e[idx_all]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), len(idx_all), int(np.sum(e[idx_all]))


def clinical_point(y, e, pred, horizons):
    mask = e == 1
    y = np.asarray(y[mask], dtype=float)
    pred = np.asarray(pred[mask], dtype=float)
    err = pred - y
    over_days = np.maximum(err, 0)
    out = {
        "n_uncensored": float(len(y)),
        "pct_predicted_after_true_tte": 100.0 * float(np.mean(err > 0)) if len(y) else np.nan,
        "mean_days_predicted_after_true_tte": float(np.mean(over_days)) if len(y) else np.nan,
    }
    for h in horizons:
        out[f"h{h}_pct_overpredicted_by_at_least_h_days"] = 100.0 * float(np.mean(err >= h)) if len(y) else np.nan
    return out


def bootstrap_clinical(y, e, pred, mask, horizons, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx_all = np.where(mask & (e == 1))[0]
    point = clinical_point(y[mask], e[mask], pred[mask], horizons)
    boot = {k: [] for k in point}
    if len(idx_all) == 0:
        return {k: (v, np.nan, np.nan) for k, v in point.items()}
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        vals = clinical_point(y[idx], np.ones(len(idx), dtype=int), pred[idx], horizons)
        for k, v in vals.items():
            if np.isfinite(v):
                boot[k].append(v)
    out = {}
    for k, v in point.items():
        vals = boot[k]
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[k] = (v, float(lo), float(hi))
        else:
            out[k] = (v, np.nan, np.nan)
    return out


def fmt(est, lo, hi, decimals):
    if not np.isfinite(est):
        return "NA & [NA, NA]"
    return f"{est:.{decimals}f} & [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_cell(est, lo, hi, decimals):
    if not np.isfinite(est):
        return "NA [NA, NA]"
    return f"{est:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def metric_table(rows, metric, decimals, label, caption):
    bins = []
    models = []
    for row in rows:
        if row["Metric"] == metric and row["Bin"] not in bins:
            bins.append(row["Bin"])
        if row["Metric"] == metric and row["Model"] not in models:
            models.append(row["Model"])
    by_key = {(r["Metric"], r["Model"], r["Bin"]): r for r in rows}
    lines = [
        "\\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        f"Model & {metric} ({bins[0]}) & {metric} ({bins[1]}) \\\\",
        "\\midrule",
    ]
    for model in models:
        cells = []
        for bin_name in bins:
            r = by_key[(metric, model, bin_name)]
            cells.append(fmt(r["estimate"], r["ci_low"], r["ci_high"], decimals))
        lines.append(f"{model} & {cells[0]} & {cells[1]} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def clinical_table(rows, models, group, horizons):
    wanted = [
        ("pct_predicted_after_true_tte", "% predicted after true TTE", 1),
        ("mean_days_predicted_after_true_tte", "Mean days predicted after true TTE", 2),
    ]
    for h in horizons:
        wanted.append((f"h{h}_pct_overpredicted_by_at_least_h_days", f"% overpredicted by at least {h}d", 1))
    by_key = {(r["Model"], r["Group"], r["Metric"]): r for r in rows}
    lines = [
        "\\begin{table}",
        f"\\caption{{Clinical overprediction metrics among uncensored test individuals: {group}.}}",
        "\\begin{tabular}{l" + "l" * len(models) + "}",
        "\\toprule",
        "Metric & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for metric, label, decimals in wanted:
        cells = []
        for model in models:
            r = by_key.get((model, group, metric))
            cells.append("NA" if r is None else fmt_cell(r["estimate"], r["ci_low"], r["ci_high"], decimals))
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_extreme_bins(y, e, preds, masks, n_boot, seed):
    rows = []
    specs = [("MAE", 2, mae_uncensored), ("C-index", 2, cindex)]
    for metric, _, fn in specs:
        for model, pred in preds.items():
            for bin_name, mask in masks.items():
                est, lo, hi, n, n_unc = bootstrap(y, e, pred, mask, fn, n_boot, seed)
                rows.append(
                    {
                        "Metric": metric,
                        "Model": model,
                        "Bin": bin_name,
                        "estimate": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                        "n_uncensored": n_unc,
                    }
                )
    return rows


def evaluate_clinical(y, e, preds, masks, horizons, n_boot, seed):
    rows = []
    for model, pred in preds.items():
        for group, mask in masks.items():
            stats = bootstrap_clinical(y, e, pred, mask, horizons, n_boot, seed)
            for metric, (est, lo, hi) in stats.items():
                rows.append({"Model": model, "Group": group, "Metric": metric, "estimate": est, "ci_low": lo, "ci_high": hi})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate final real-data MAE, C-index, clinical, and clustering sensitivity tables.")
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/final_realdata_report_tables")
    parser.add_argument("--model", action="append", default=[], help="Main model as Name=/path/to/pred.joblib")
    parser.add_argument("--cluster-model", action="append", default=[], help="Cluster sensitivity model as Name=/path/to/pred.joblib")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 7])
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-defaults", action="store_true", help="Require explicit --model/--cluster-model paths.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = load_split(args.data_dir, "test", "y")
    e = load_split(args.data_dir, "test", "binary_y").astype(int)
    propensity, best_c, val_auroc = fit_propensity(args.data_dir)
    masks = make_masks(propensity, args.low_max, args.high_min)
    clinical_masks = make_clinical_masks(propensity, args.low_max, args.high_min)

    main_paths = parse_named_paths(args.model, [] if args.no_defaults else DEFAULT_MODELS)
    cluster_paths = parse_named_paths(args.cluster_model, [] if args.no_defaults else DEFAULT_CLUSTER_MODELS)
    main_preds = load_predictions(main_paths, len(y))
    cluster_preds = load_predictions(cluster_paths, len(y))

    print(f"Propensity model: best C={best_c}, val AUROC={val_auroc:.4f}", flush=True)
    for name, mask in clinical_masks.items():
        print(f"{name}: n={int(mask.sum())}, uncensored={int(e[mask].sum())}", flush=True)

    main_rows = evaluate_extreme_bins(y, e, main_preds, masks, args.n_boot, args.seed)
    cluster_rows = evaluate_extreme_bins(y, e, cluster_preds, masks, args.n_boot, args.seed)
    clinical_rows = evaluate_clinical(y, e, main_preds, clinical_masks, args.horizons, args.n_boot, args.seed)

    write_csv(
        out_dir / "main_extreme_bin_metrics.csv",
        main_rows,
        ["Metric", "Model", "Bin", "estimate", "ci_low", "ci_high", "n", "n_uncensored"],
    )
    write_csv(
        out_dir / "cluster_sensitivity_extreme_bin_metrics.csv",
        cluster_rows,
        ["Metric", "Model", "Bin", "estimate", "ci_low", "ci_high", "n", "n_uncensored"],
    )
    write_csv(
        out_dir / "clinical_overprediction_metrics.csv",
        clinical_rows,
        ["Model", "Group", "Metric", "estimate", "ci_low", "ci_high"],
    )

    main_tex = "\n\n".join(
        [
            metric_table(main_rows, "MAE", 2, "tab:realdata_mae_extreme_bins", "MAE with 95\\% confidence intervals for real-data propensity subgroups."),
            metric_table(main_rows, "C-index", 2, "tab:realdata_cindex_extreme_bins", "C-index with 95\\% confidence intervals for real-data propensity subgroups."),
        ]
    )
    cluster_tex = "\n\n".join(
        [
            metric_table(cluster_rows, "MAE", 2, "tab:realdata_cluster_sensitivity_mae", "Clustering sensitivity: MAE with 95\\% confidence intervals."),
            metric_table(cluster_rows, "C-index", 2, "tab:realdata_cluster_sensitivity_cindex", "Clustering sensitivity: C-index with 95\\% confidence intervals."),
        ]
    )
    clinical_tex = "\n\n".join(
        clinical_table(clinical_rows, list(main_preds.keys()), group, args.horizons)
        for group in clinical_masks
    )

    (out_dir / "main_extreme_bin_tables.tex").write_text(main_tex + "\n")
    (out_dir / "cluster_sensitivity_tables.tex").write_text(cluster_tex + "\n")
    (out_dir / "clinical_overprediction_tables.tex").write_text(clinical_tex + "\n")
    (out_dir / "all_report_tables.tex").write_text(main_tex + "\n\n" + clinical_tex + "\n\n" + cluster_tex + "\n")

    print(f"Saved outputs to {out_dir}", flush=True)
    print("\n" + main_tex, flush=True)
    print("\n" + clinical_table(clinical_rows, list(main_preds.keys()), "All test individuals", args.horizons), flush=True)
    print("\n" + cluster_tex, flush=True)


if __name__ == "__main__":
    main()
