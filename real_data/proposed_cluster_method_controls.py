import argparse
import csv
import json
import os
import time
from pathlib import Path
from cwite_paths import real_data_dir, real_run_path, real_output_path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid

import realdata_overall_sweep as sweep


def assign_nearest_centroid(x_train, train_clusters, x_other):
    clf = NearestCentroid()
    clf.fit(x_train, train_clusters)
    return clf.predict(x_other)


def cluster_assignments(config, x_train, x_val, x_test):
    method = config["cluster_method"]
    k = int(config["k"])
    seed = int(config["seed"])
    start = time.time()
    sweep.log(f"Fitting cluster method={method}, k={k}")

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        train_clusters = model.fit_predict(x_train)
        val_clusters = model.predict(x_val)
        test_clusters = model.predict(x_test)

    elif method == "kmeans_pca":
        n_components = min(int(config["cluster_pca_dim"]), x_train.shape[1], x_train.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=seed)
        z_train = pca.fit_transform(x_train)
        z_val = pca.transform(x_val)
        z_test = pca.transform(x_test)
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        train_clusters = model.fit_predict(z_train)
        val_clusters = model.predict(z_val)
        test_clusters = model.predict(z_test)
        sweep.log(f"  kmeans_pca n_components={n_components}, explained_var={np.sum(pca.explained_variance_ratio_):.3f}")

    elif method == "gmm":
        model = GaussianMixture(n_components=k, random_state=seed, covariance_type="diag", reg_covar=1e-6)
        train_clusters = model.fit_predict(x_train)
        val_clusters = model.predict(x_val)
        test_clusters = model.predict(x_test)

    elif method == "spectral":
        # sklearn SpectralClustering has no out-of-sample predict. Fit train labels, then
        # assign val/test to nearest train-label centroid in the same feature space.
        model = SpectralClustering(
            n_clusters=k,
            random_state=seed,
            affinity="nearest_neighbors",
            n_neighbors=min(10, max(1, x_train.shape[0] - 1)),
            assign_labels="kmeans",
        )
        train_clusters = model.fit_predict(x_train)
        val_clusters = assign_nearest_centroid(x_train, train_clusters, x_val)
        test_clusters = assign_nearest_centroid(x_train, train_clusters, x_test)

    elif method == "random":
        rng = np.random.default_rng(int(config.get("random_cluster_seed", seed)))
        train_clusters = rng.integers(0, k, size=x_train.shape[0])
        val_clusters = rng.integers(0, k, size=x_val.shape[0])
        test_clusters = rng.integers(0, k, size=x_test.shape[0])

    elif method == "global":
        train_clusters = np.zeros(x_train.shape[0], dtype=int)
        val_clusters = np.zeros(x_val.shape[0], dtype=int)
        test_clusters = np.zeros(x_test.shape[0], dtype=int)

    else:
        raise ValueError(f"Unknown cluster_method={method}")

    unique, counts = np.unique(train_clusters, return_counts=True)
    sweep.log(
        f"Cluster method={method} complete: clusters={len(unique)}, "
        f"min_size={int(np.min(counts))}, median_size={np.median(counts):.1f}, "
        f"max_size={int(np.max(counts))} ({sweep.format_seconds(time.time() - start)})"
    )
    return train_clusters, val_clusters, test_clusters


def method_cluster_weights(config, x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test):
    if config["method"] != "Proposed":
        return np.ones(len(y_train)), np.ones(len(y_val)), np.ones(len(y_test))

    train_clusters, val_clusters, test_clusters = cluster_assignments(config, x_train, x_val, x_test)
    weights = (
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, train_clusters, y_train, e_train, config["weight_func"]),
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, val_clusters, y_val, e_val, config["weight_func"]),
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, test_clusters, y_test, e_test, config["weight_func"]),
    )
    sweep.log(
        f"Cluster weights: train_censored_mean={np.mean(weights[0][e_train == 0]):.3f}, "
        f"val_censored_mean={np.mean(weights[1][e_val == 0]):.3f}, "
        f"test_censored_mean={np.mean(weights[2][e_test == 0]):.3f}"
    )
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Retrain a selected CWITE config once per clustering method for real-data sensitivity analysis."
    )
    parser.add_argument("--config", required=True, help="Path to proposed_best_config.json from the selected CWITE run.")
    parser.add_argument("--data-dir", default=real_data_dir())
    parser.add_argument("--out-dir", default=real_output_path('proposed_cluster_method_controls'))
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument(
        "--cluster-methods",
        nargs="+",
        choices=["kmeans", "kmeans_pca", "gmm", "spectral", "random", "global"],
        default=["kmeans", "kmeans_pca", "gmm", "spectral"],
    )
    parser.add_argument("--cluster-pca-dim", type=int, default=20)
    parser.add_argument("--random-cluster-seed", type=int, default=123)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--low-max", type=float, default=0.10)
    parser.add_argument("--high-min", type=float, default=0.70)
    parser.add_argument("--high-penalty", type=float, default=0.25)
    parser.add_argument("--prediction-mode", choices=["expected", "argmax"], default="argmax")
    parser.add_argument("--selection-score", choices=["overall", "low_high", "low_only"], default="overall")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_dir / "proposed_cluster_method_controls.log.txt"
    sweep.set_log_file(log_path)
    sweep.acquire_out_dir_lock(out_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = json.loads(Path(args.config).read_text())
    base_config["method"] = "Proposed"
    base_config.setdefault("proposed_loss_scaling", "normalized")
    base_config["cluster_pca_dim"] = args.cluster_pca_dim
    base_config["random_cluster_seed"] = args.random_cluster_seed

    train_args = SimpleNamespace(
        max_epochs=args.max_epochs,
        patience=args.patience,
        low_max=args.low_max,
        high_min=args.high_min,
        high_penalty=args.high_penalty,
        prediction_mode=args.prediction_mode,
        selection_score=args.selection_score,
        log_every_epochs=args.log_every_epochs,
    )

    sweep.log(f"Arguments: {json.dumps(vars(args), sort_keys=True)}")
    sweep.log(f"Loaded selected CWITE config: {json.dumps(base_config, sort_keys=True)}")
    sweep.log(f"Using device={device}; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    sweep.log("This is NOT a hyperparameter sweep: selected CWITE config is fixed; only cluster method changes.")

    x_train_raw, y_train, e_train = sweep.load_split(args.data_dir, "train")
    x_val_raw, y_val, e_val = sweep.load_split(args.data_dir, "val")
    x_test_raw, y_test, e_test = sweep.load_split(args.data_dir, "test")
    prop_train, prop_val, prop_test, best_c, val_auroc = sweep.fit_propensity(
        x_train_raw, e_train, x_val_raw, e_val, x_test_raw
    )
    sweep.log(f"Propensity best C={best_c}, val AUROC={val_auroc:.4f}")

    x_train, x_val, x_test = sweep.transform_features(
        base_config["transform"], int(base_config["dim"]), x_train_raw, y_train, e_train, x_val_raw, x_test_raw
    )

    original_make_cluster_weights = sweep.make_cluster_weights
    sweep.make_cluster_weights = method_cluster_weights
    rows = []
    try:
        for method in args.cluster_methods:
            config = dict(base_config)
            config["cluster_method"] = method
            config["status"] = f"{method}_cluster_method_control"
            start = time.time()
            sweep.log("=" * 80)
            sweep.log(f"CLUSTER METHOD START: {method}")
            result = sweep.train_one(
                config,
                (x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test),
                (prop_train, prop_val, prop_test),
                device,
                train_args,
            )
            row = {
                **config,
                "val_low_mae": result["val_low_mae"],
                "val_high_mae": result["val_high_mae"],
                "val_overall_mae": result["val_overall_mae"],
                "val_score": result["val_score"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "seconds": round(time.time() - start, 3),
            }
            rows.append(row)
            pred_path = out_dir / f"cwite_{method}_y_test_pred.joblib"
            config_path = out_dir / f"cwite_{method}_config.json"
            joblib.dump(result["test_pred"], pred_path)
            config_path.write_text(json.dumps(row, indent=2, sort_keys=True))
            sweep.log(f"CLUSTER METHOD DONE: {method}; saved {pred_path} and {config_path}")
    finally:
        sweep.make_cluster_weights = original_make_cluster_weights

    csv_path = out_dir / "cluster_method_control_results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        sweep.log(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if sweep.LOG_FH is not None:
            sweep.LOG_FH.flush()
            os.fsync(sweep.LOG_FH.fileno())
            sweep.LOG_FH.close()
        if sweep.LOCK_FH is not None:
            import fcntl

            fcntl.flock(sweep.LOCK_FH.fileno(), fcntl.LOCK_UN)
