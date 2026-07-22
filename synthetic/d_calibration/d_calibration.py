import argparse
from pathlib import Path

import joblib
import numpy as np


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_discrete_time_logistic(X, y, num_times, lr=0.05, epochs=2000, weight_decay=1e-4, seed=0):
    """Fit a simple multinomial model P(T=t | X) using numpy only."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).clip(0, num_times - 1)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    X_design = np.c_[np.ones(len(X_scaled)), X_scaled]

    weights = rng.normal(scale=0.01, size=(X_design.shape[1], num_times))
    eye_y = np.eye(num_times)[y]

    for _ in range(epochs):
        probs = softmax(X_design @ weights)
        grad = X_design.T @ (probs - eye_y) / len(X_design)
        grad[1:] += weight_decay * weights[1:]
        weights -= lr * grad

    return weights, X_mean, X_std


def predict_pmf(X, weights, X_mean, X_std):
    X = np.asarray(X, dtype=float)
    X_scaled = (X - X_mean) / X_std
    X_design = np.c_[np.ones(len(X_scaled)), X_scaled]
    return softmax(X_design @ weights)


def survival_at_times(pmf, times):
    """Return P(T > time) for each row of a discrete event-time PMF."""
    times = np.asarray(times, dtype=int).clip(0, pmf.shape[1] - 1)
    cdf = np.cumsum(pmf, axis=1)
    return 1.0 - cdf[np.arange(len(times)), times]


def add_uniform_interval_mass(bin_masses, bin_edges, left, right, total_mass=1.0):
    left = float(np.clip(left, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    if right < left:
        left, right = right, left
    width = right - left

    if width <= 0.0:
        bin_idx = min(np.searchsorted(bin_edges, right, side="right") - 1, len(bin_masses) - 1)
        bin_masses[max(bin_idx, 0)] += total_mass
        return

    for bin_idx in range(len(bin_masses)):
        bin_left = bin_edges[bin_idx]
        bin_right = bin_edges[bin_idx + 1]
        overlap = max(0.0, min(bin_right, right) - max(bin_left, left))
        bin_masses[bin_idx] += total_mass * overlap / width


def d_calibration(event_times, event_indicators, pmf, num_bins=10):
    """
    Compute D-calibration bin masses.

    For discrete-time predictions, uncensored observations contribute one count
    spread uniformly over the survival interval [S(T_i), S(T_i - 1)]. Censored
    observations at C_i contribute one count spread uniformly over [0, S(C_i)].
    """
    event_times = np.asarray(event_times)
    event_indicators = np.asarray(event_indicators).astype(bool)
    pmf = np.asarray(pmf, dtype=float)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)

    survival_probs = survival_at_times(pmf, event_times)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_masses = np.zeros(num_bins)

    for row_idx, (s_i, observed) in enumerate(zip(survival_probs, event_indicators)):
        s_i = float(np.clip(s_i, 0.0, 1.0))
        if observed:
            event_time = int(np.clip(event_times[row_idx], 0, pmf.shape[1] - 1))
            interval_right = min(1.0, s_i + pmf[row_idx, event_time])
            add_uniform_interval_mass(bin_masses, bin_edges, s_i, interval_right)
        elif s_i > 0.0:
            add_uniform_interval_mass(bin_masses, bin_edges, 0.0, s_i)

    expected = len(event_times) / num_bins
    chi_square = np.sum((bin_masses - expected) ** 2 / expected)

    try:
        from scipy.stats import chi2

        p_value = float(chi2.sf(chi_square, df=num_bins - 1))
    except ImportError:
        p_value = np.nan

    return {
        "bin_edges": bin_edges,
        "bin_masses": bin_masses,
        "expected_per_bin": expected,
        "chi_square": float(chi_square),
        "p_value": p_value,
    }


def load_pmf(path_template, count):
    path = Path(path_template.format(count=count, suffix=f"COUNT{count}"))
    if path.suffix == ".npy":
        return np.load(path)
    return joblib.load(path)


def format_pmf_path(path_template, model, count):
    return path_template.format(model=model, count=count, suffix=f"COUNT{count}")


def pmf_exists(path_template, model, count):
    path = Path(format_pmf_path(path_template, model, count))
    return path.exists()


def main():
    parser = argparse.ArgumentParser(description="Compute D-calibration on existing synthetic survival data.")
    parser.add_argument("--data-dir", default="/data4/meerak/onevar_data")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--idxmin", type=int, default=None)
    parser.add_argument("--idxmax", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pmf-path", default=None, help="Optional joblib/npy array with shape (n_test, num_times).")
    parser.add_argument(
        "--pmf-template",
        default="/data4/meerak/onevar_test_pmfs/{model}_y_test_pmf_{suffix}.joblib",
        help="Template for method PMF files. Available fields: {model}, {count}, {suffix}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["icind", "pcind", "dhcind", "proposed", "oracle"],
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.count is not None:
        counts = [args.count]
    elif args.idxmin is not None and args.idxmax is not None:
        counts = range(args.idxmin, args.idxmax)
    else:
        counts = [0]

    all_results = []
    for count in counts:
        suffix = f"COUNT{count}"

        X_train = joblib.load(data_dir / f"X_train_{suffix}.joblib").reshape(-1, 1)
        y_train = joblib.load(data_dir / f"y_train_{suffix}.joblib")
        binary_y_train = joblib.load(data_dir / f"binary_y_train_{suffix}.joblib")

        X_test = joblib.load(data_dir / f"X_test_{suffix}.joblib").reshape(-1, 1)
        y_test = joblib.load(data_dir / f"y_test_{suffix}.joblib")
        binary_y_test = joblib.load(data_dir / f"actual_binary_y_test{suffix}.joblib")

        num_times = int(max(np.max(y_train), np.max(y_test))) + 1

        if args.pmf_path is not None:
            test_pmf = load_pmf(args.pmf_path, count)
            results = d_calibration(y_test, binary_y_test, test_pmf, num_bins=args.num_bins)
            all_results.append((count, "pmf_path", results))

            print(f"Dataset: {suffix}")
            print(f"Model: pmf_path")
            print(f"n_test: {len(y_test)}")
            print(f"bin_edges: {np.round(results['bin_edges'], 3).tolist()}")
            print(f"bin_masses: {np.round(results['bin_masses'], 3).tolist()}")
            print(f"expected_per_bin: {results['expected_per_bin']:.3f}")
            print(f"chi_square: {results['chi_square']:.3f}")
            if np.isnan(results["p_value"]):
                print("p_value: nan (install scipy to compute it)")
            else:
                print(f"p_value: {results['p_value']:.6f}")
            print()
        elif args.pmf_template is not None and all(pmf_exists(args.pmf_template, model, count) for model in args.models):
            for model in args.models:
                test_pmf = load_pmf(format_pmf_path(args.pmf_template, model, count), count)
                results = d_calibration(y_test, binary_y_test, test_pmf, num_bins=args.num_bins)
                all_results.append((count, model, results))

                print(f"Dataset: {suffix}")
                print(f"Model: {model}")
                print(f"n_test: {len(y_test)}")
                print(f"bin_edges: {np.round(results['bin_edges'], 3).tolist()}")
                print(f"bin_masses: {np.round(results['bin_masses'], 3).tolist()}")
                print(f"expected_per_bin: {results['expected_per_bin']:.3f}")
                print(f"chi_square: {results['chi_square']:.3f}")
                if np.isnan(results["p_value"]):
                    print("p_value: nan (install scipy to compute it)")
                else:
                    print(f"p_value: {results['p_value']:.6f}")
                print()
        elif args.pmf_template is not None:
            print(f"Dataset: {suffix}")
            print("Skipped: missing at least one method PMF file")
            print()
        else:
            observed = binary_y_train.astype(bool)
            weights, X_mean, X_std = fit_discrete_time_logistic(
                X_train[observed],
                y_train[observed],
                num_times=num_times,
                lr=args.lr,
                epochs=args.epochs,
            )
            test_pmf = predict_pmf(X_test, weights, X_mean, X_std)
            results = d_calibration(y_test, binary_y_test, test_pmf, num_bins=args.num_bins)
            all_results.append((count, "fitted_logistic", results))

            print(f"Dataset: {suffix}")
            print("Model: fitted_logistic")
            print(f"n_test: {len(y_test)}")
            print(f"bin_edges: {np.round(results['bin_edges'], 3).tolist()}")
            print(f"bin_masses: {np.round(results['bin_masses'], 3).tolist()}")
            print(f"expected_per_bin: {results['expected_per_bin']:.3f}")
            print(f"chi_square: {results['chi_square']:.3f}")
            if np.isnan(results["p_value"]):
                print("p_value: nan (install scipy to compute it)")
            else:
                print(f"p_value: {results['p_value']:.6f}")
            print()

    if len(all_results) > 1:
        print(f"included_results: {len(all_results)}")
        for model in sorted({model for _, model, _ in all_results}):
            model_results = [results for _, result_model, results in all_results if result_model == model]
            chi_squares = [results["chi_square"] for results in model_results]
            p_values = [results["p_value"] for results in model_results]
            print(f"{model} mean_chi_square: {np.mean(chi_squares):.3f}")
            if not np.all(np.isnan(p_values)):
                print(f"{model} mean_p_value: {np.nanmean(p_values):.6f}")


if __name__ == "__main__":
    main()
