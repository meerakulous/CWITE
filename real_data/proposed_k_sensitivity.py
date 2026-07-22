import argparse
import csv
import json
import os
import time
from pathlib import Path
from cwite_paths import real_data_dir, real_run_path, real_output_path
from types import SimpleNamespace

import joblib
import torch

import realdata_overall_sweep as sweep


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the selected CWITE config once per k to assess sensitivity to "
            "the number of k-means clusters. All non-k hyperparameters are fixed."
        )
    )
    parser.add_argument("--config", required=True, help="Path to selected cwite/proposed_best_config.json.")
    parser.add_argument("--data-dir", default=real_data_dir())
    parser.add_argument("--out-dir", default=real_run_path('cwite_k_sensitivity'))
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 5, 8, 10, 15, 20, 30, 40])
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
    log_path = Path(args.log_file) if args.log_file else out_dir / "cwite_k_sensitivity.log.txt"
    sweep.set_log_file(log_path)
    sweep.acquire_out_dir_lock(out_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = json.loads(Path(args.config).read_text())
    base_config["method"] = "Proposed"
    base_config.setdefault("proposed_loss_scaling", "normalized")

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
    sweep.log("This is NOT a hyperparameter sweep: selected CWITE config is fixed; only k changes.")

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

    rows = []
    for k in args.k_values:
        config = dict(base_config)
        config["k"] = int(k)
        config["status"] = "k_sensitivity"
        start = time.time()
        sweep.log("=" * 80)
        sweep.log(f"K SENSITIVITY START: k={k}")
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
        pred_path = out_dir / f"cwite_k{k}_y_test_pred.joblib"
        config_path = out_dir / f"cwite_k{k}_config.json"
        joblib.dump(result["test_pred"], pred_path)
        config_path.write_text(json.dumps(row, indent=2, sort_keys=True))
        sweep.log(f"K SENSITIVITY DONE: k={k}; saved {pred_path} and {config_path}")

    csv_path = out_dir / "cwite_k_sensitivity_results.csv"
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
