import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

import realdata_overall_sweep as sweep


def control_cluster_weights(config, x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test):
    if config["method"] != "Proposed":
        return np.ones(len(y_train)), np.ones(len(y_val)), np.ones(len(y_test))

    start = time.time()
    seed = int(config.get("random_cluster_seed", config["seed"]))
    k = int(config["k"])
    mode = config["cluster_control"]
    sweep.log(
        f"Assigning {mode.upper()} clusters for Proposed control: k={k}, "
        f"weight_func={config['weight_func']}, random_cluster_seed={seed}"
    )
    if mode == "random":
        rng = np.random.default_rng(seed)
        train_clusters = rng.integers(0, k, size=len(y_train))
        val_clusters = rng.integers(0, k, size=len(y_val))
        test_clusters = rng.integers(0, k, size=len(y_test))
    elif mode == "global":
        train_clusters = np.zeros(len(y_train), dtype=int)
        val_clusters = np.zeros(len(y_val), dtype=int)
        test_clusters = np.zeros(len(y_test), dtype=int)
    else:
        raise ValueError(f"Unknown cluster_control={mode}")

    weights = (
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, train_clusters, y_train, e_train, config["weight_func"]),
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, val_clusters, y_val, e_val, config["weight_func"]),
        sweep.cluster_reference_weights(y_train, e_train, train_clusters, test_clusters, y_test, e_test, config["weight_func"]),
    )
    unique, counts = np.unique(train_clusters, return_counts=True)
    sweep.log(
        f"{mode.capitalize()} clusters complete: clusters={len(unique)}, min_size={int(np.min(counts))}, "
        f"median_size={float(np.median(counts)):.1f}, max_size={int(np.max(counts))}, "
        f"train_censored_weight_mean={float(np.mean(weights[0][e_train == 0])):.3f} "
        f"({sweep.format_seconds(time.time() - start)})"
    )
    return weights


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the already-selected Proposed/CWITE config once with altered clusters, "
            "to test whether KMeans cluster structure is carrying the gain."
        )
    )
    parser.add_argument("--config", required=True, help="Path to proposed_best_config.json from the KMeans sweep.")
    parser.add_argument("--data-dir", default="/data4/meerak/real_labor")
    parser.add_argument("--out-dir", default="/data4/meerak/proposed_random_cluster_control")
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("--cluster-control", choices=["random", "global"], default="random")
    parser.add_argument("--random-cluster-seed", type=int, default=123)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
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
    log_path = Path(args.log_file) if args.log_file else out_dir / "proposed_random_cluster_control.log.txt"
    sweep.set_log_file(log_path)
    sweep.acquire_out_dir_lock(out_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = json.loads(Path(args.config).read_text())
    config["method"] = "Proposed"
    config["random_cluster_seed"] = args.random_cluster_seed
    config["cluster_control"] = args.cluster_control
    config["status"] = f"{args.cluster_control}_cluster_control"
    config.setdefault("proposed_loss_scaling", "normalized")

    train_args = SimpleNamespace(
        max_epochs=args.max_epochs if args.max_epochs is not None else int(config.get("max_epochs", 200)),
        patience=args.patience if args.patience is not None else int(config.get("patience", 15)),
        low_max=args.low_max,
        high_min=args.high_min,
        high_penalty=args.high_penalty,
        prediction_mode=args.prediction_mode,
        selection_score=args.selection_score,
        log_every_epochs=args.log_every_epochs,
    )

    sweep.log(f"Arguments: {json.dumps(vars(args), sort_keys=True)}")
    sweep.log(f"Loaded selected Proposed config: {json.dumps(config, sort_keys=True)}")
    sweep.log(f"Using device={device}; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    sweep.log(
        "This is NOT a sweep: the config is fixed; only KMeans clusters are replaced by "
        f"{args.cluster_control} clusters."
    )

    x_train_raw, y_train, e_train = sweep.load_split(args.data_dir, "train")
    x_val_raw, y_val, e_val = sweep.load_split(args.data_dir, "val")
    x_test_raw, y_test, e_test = sweep.load_split(args.data_dir, "test")
    prop_train, prop_val, prop_test, best_c, val_auroc = sweep.fit_propensity(
        x_train_raw, e_train, x_val_raw, e_val, x_test_raw
    )
    sweep.log(f"Propensity best C={best_c}, val AUROC={val_auroc:.4f}")

    x_train, x_val, x_test = sweep.transform_features(
        config["transform"], int(config["dim"]), x_train_raw, y_train, e_train, x_val_raw, x_test_raw
    )

    original_make_cluster_weights = sweep.make_cluster_weights
    sweep.make_cluster_weights = control_cluster_weights
    try:
        result = sweep.train_one(
            config,
            (x_train, y_train, e_train, x_val, y_val, e_val, x_test, y_test, e_test),
            (prop_train, prop_val, prop_test),
            device,
            train_args,
        )
    finally:
        sweep.make_cluster_weights = original_make_cluster_weights

    row = {
        **config,
        "cluster_control": args.cluster_control,
        "val_low_mae": result["val_low_mae"],
        "val_high_mae": result["val_high_mae"],
        "val_overall_mae": result["val_overall_mae"],
        "val_score": result["val_score"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
    }

    pred_path = out_dir / f"{args.cluster_control}_cluster_proposed_y_test_pred.joblib"
    config_path = out_dir / f"{args.cluster_control}_cluster_proposed_config.json"
    joblib.dump(result["test_pred"], pred_path)
    config_path.write_text(json.dumps(row, indent=2, sort_keys=True))
    sweep.log(f"Saved random-cluster predictions: {pred_path}")
    sweep.log(f"Saved random-cluster config: {config_path}")


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
