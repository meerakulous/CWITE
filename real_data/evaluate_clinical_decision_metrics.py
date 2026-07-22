import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_real_labor_split(data_dir, split, name):
    return joblib.load(Path(data_dir) / f"real_labor_{name}_{split}.joblib")


def fit_propensity_model(data_dir):
    x_train = load_real_labor_split(data_dir, "train", "X")
    e_train = load_real_labor_split(data_dir, "train", "binary_y")
    x_val = load_real_labor_split(data_dir, "val", "X")
    e_val = load_real_labor_split(data_dir, "val", "binary_y")
    x_test = load_real_labor_split(data_dir, "test", "X")

    cs = [1, 1e-2, 1e-4, 1e-6]
    val_aurocs = []
    for c in cs:
        clf = LogisticRegression(random_state=0, C=c, max_iter=1000).fit(x_train, e_train)
        val_aurocs.append(roc_auc_score(e_val, clf.predict_proba(x_val)[:, 1]))

    best_c = cs[int(np.argmax(val_aurocs))]
    clf = LogisticRegression(random_state=0, C=best_c, max_iter=1000).fit(x_train, e_train)
    return clf.predict_proba(x_test)[:, 1], best_c, max(val_aurocs)


def parse_model_args(model_args):
    models = {}
    for item in model_args:
        if "=" not in item:
            raise ValueError(f"Bad --model value {item!r}; expected Name=prediction_path")
        name, path = item.split("=", 1)
        models[name] = path
    if not models:
        raise ValueError("Pass at least one --model Name=prediction_path")
    return models


def load_predictions(models):
    preds = {}
    for name, path in models.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing predictions for {name}: {p}")
        preds[name] = np.asarray(joblib.load(p)).reshape(-1)
    return preds


def make_masks(propensity, low_max, high_min):
    return {
        "All test individuals": np.ones(len(propensity), dtype=bool),
        f"Censored-like: Propensity [0.00, {low_max:.2f})": propensity < low_max,
        f"Uncensored-like: Propensity [{high_min:.2f}, 1.00]": propensity >= high_min,
    }


def pct(x):
    return 100.0 * float(x)


def safe_mean(x):
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_median(x):
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def compute_metrics(y, e, pred, mask, horizons):
    eval_mask = mask & (e == 1)
    y_u = np.asarray(y[eval_mask], dtype=float)
    pred_u = np.asarray(pred[eval_mask], dtype=float)
    err = pred_u - y_u
    over = err > 0
    under = err < 0
    abs_err = np.abs(err)
    over_days = np.maximum(err, 0)

    rows = [
        ("n_uncensored", float(len(y_u))),
        ("mae_days", safe_mean(abs_err)),
        ("median_abs_error_days", safe_median(abs_err)),
        ("pct_predicted_after_true_tte", pct(np.mean(over)) if len(y_u) else np.nan),
        ("pct_predicted_before_true_tte", pct(np.mean(under)) if len(y_u) else np.nan),
        ("mean_days_predicted_after_true_tte", safe_mean(over_days)),
        ("median_days_predicted_after_true_tte", safe_median(over_days)),
    ]

    for h in horizons:
        over_by_h = err >= h
        true_within = y_u <= h
        pred_within = pred_u <= h
        n_true = int(np.sum(true_within))
        n_pred = int(np.sum(pred_within))
        caught = true_within & pred_within
        missed = true_within & (~pred_within)

        rows.extend(
            [
                (f"h{h}_pct_overpredicted_by_at_least_h_days", pct(np.mean(over_by_h)) if len(y_u) else np.nan),
                (f"h{h}_pct_true_spontaneous_within_h_days", pct(np.mean(true_within)) if len(y_u) else np.nan),
                (f"h{h}_pct_model_recommends_wait", pct(np.mean(pred_within)) if len(y_u) else np.nan),
                (f"h{h}_pct_true_within_h_caught_by_model", pct(np.sum(caught) / n_true) if n_true else np.nan),
                (f"h{h}_pct_true_within_h_missed_by_model", pct(np.sum(missed) / n_true) if n_true else np.nan),
                (f"h{h}_ppv_among_wait_recommended", pct(np.sum(caught) / n_pred) if n_pred else np.nan),
                (f"h{h}_n_true_within_h", float(n_true)),
                (f"h{h}_n_wait_recommended", float(n_pred)),
                (f"h{h}_n_true_within_h_caught", float(np.sum(caught))),
                (f"h{h}_n_true_within_h_missed", float(np.sum(missed))),
            ]
        )

    return dict(rows)


def bootstrap_metrics(y, e, pred, mask, horizons, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = np.where(mask & (e == 1))[0]
    point = compute_metrics(y, e, pred, mask, horizons)
    boot = {metric: [] for metric in point}

    if len(idx) == 0:
        return {
            metric: {"estimate": val, "ci_low": np.nan, "ci_high": np.nan}
            for metric, val in point.items()
        }

    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        # Work on sampled arrays directly so repeated bootstrap rows are represented.
        vals = compute_metrics(y[sample], np.ones(len(sample), dtype=int), pred[sample], np.ones(len(sample), dtype=bool), horizons)
        for metric, val in vals.items():
            if np.isfinite(val):
                boot[metric].append(val)

    out = {}
    for metric, val in point.items():
        vals = boot[metric]
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[metric] = {"estimate": val, "ci_low": float(lo), "ci_high": float(hi)}
        else:
            out[metric] = {"estimate": val, "ci_low": np.nan, "ci_high": np.nan}
    return out


def fmt(est, lo, hi, decimals=1):
    if not np.isfinite(est):
        return "NA [NA, NA]"
    return f"{est:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def latex_summary(rows, models, groups, horizons):
    wanted = [
        ("pct_predicted_after_true_tte", "% predicted after true TTE"),
        ("mean_days_predicted_after_true_tte", "Mean days predicted after true TTE"),
    ]
    for h in horizons:
        wanted.extend(
            [
                (f"h{h}_pct_overpredicted_by_at_least_h_days", f"% overpredicted by at least {h}d"),
                (f"h{h}_pct_true_within_h_caught_by_model", f"% true labor within {h}d caught"),
                (f"h{h}_pct_true_within_h_missed_by_model", f"% true labor within {h}d missed"),
                (f"h{h}_ppv_among_wait_recommended", f"PPV if wait {h}d recommended"),
            ]
        )

    by_key = {(r["Model"], r["Group"], r["Metric"]): r for r in rows}
    tables = []
    for group in groups:
        lines = [
            "\\begin{table}",
            f"\\caption{{Clinical interpretability metrics among uncensored test individuals: {group}.}}",
            "\\begin{tabular}{l" + "l" * len(models) + "}",
            "\\toprule",
            "Metric & " + " & ".join(models) + " \\\\",
            "\\midrule",
        ]
        for metric, label in wanted:
            pieces = []
            for model in models:
                r = by_key.get((model, group, metric))
                if r is None:
                    pieces.append("NA")
                else:
                    decimals = 2 if metric.startswith("mean_days") else 1
                    pieces.append(fmt(r["estimate"], r["ci_low"], r["ci_high"], decimals))
            lines.append(label + " & " + " & ".join(pieces) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute reviewer-facing clinical interpretation metrics from TTE predictions: "
            "overprediction and wait-horizon decision summaries among uncensored test individuals."
        )
    )
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/clinical_decision_metrics")
    parser.add_argument("--model", action="append", default=[], help="Model as Name=prediction_path")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 7])
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_test = load_real_labor_split(args.data_dir, "test", "y")
    e_test = load_real_labor_split(args.data_dir, "test", "binary_y").astype(int)
    propensity, best_c, val_auroc = fit_propensity_model(args.data_dir)
    models = parse_model_args(args.model)
    preds = load_predictions(models)

    n = len(y_test)
    for model, pred in preds.items():
        if len(pred) != n:
            raise ValueError(f"{model} has {len(pred)} predictions but y_test has {n} rows")

    masks = make_masks(propensity, args.low_max, args.high_min)
    print(f"Propensity model: best C={best_c}, val AUROC={val_auroc:.4f}", flush=True)
    for group, mask in masks.items():
        print(
            f"{group}: n={int(mask.sum())}, uncensored={int(e_test[mask].sum())}, "
            f"uncensored_rate={float(np.mean(e_test[mask])):.3f}",
            flush=True,
        )

    rows = []
    for model, pred in preds.items():
        print(f"Evaluating {model}", flush=True)
        for group, mask in masks.items():
            stats = bootstrap_metrics(y_test, e_test, pred, mask, args.horizons, args.n_boot, args.seed)
            for metric, values in stats.items():
                rows.append(
                    {
                        "Model": model,
                        "Group": group,
                        "Metric": metric,
                        "estimate": values["estimate"],
                        "ci_low": values["ci_low"],
                        "ci_high": values["ci_high"],
                    }
                )

    csv_path = out_dir / "clinical_decision_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Group", "Metric", "estimate", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "clinical_decision_metrics_tables.tex"
    tex = latex_summary(rows, list(models.keys()), list(masks.keys()), args.horizons)
    tex_path.write_text(tex + "\n")

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved LaTeX: {tex_path}", flush=True)
    print("\n" + tex, flush=True)


if __name__ == "__main__":
    main()
