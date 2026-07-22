# Hyperparameter Recovery: real

Archive: `<archive-directory>/real_labor_clean.tar.gz`

## Paper Table Rows

No selected hyperparameter rows were found in this archive. If this is a code-only archive, rerun on the result/output tarball containing `*_best_config.json`, `sweep_results.csv`, or legacy `*.txt` files.

## Recovered Search Metadata

### `real_labor_clean/proposed_cluster_method_controls.py`
- No grid metadata pattern detected.

### `real_labor_clean/proposed_k_sensitivity.py`
- No grid metadata pattern detected.

### `real_labor_clean/proposed_random_cluster_control.py`
- No grid metadata pattern detected.

### `real_labor_clean/realdata_overall_sweep.py`
- `optimizer`: `Adam`

## Real-Data Runner Commands

### `cwite_cluster_methods`
- `cluster_methods`: `['kmeans', 'kmeans_pca', 'gmm', 'spectral']`
- `cluster_pca_dim`: `20`
- `config`: `RUN_ROOT/cwite/proposed_best_config.json`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `out_dir`: `RUN_ROOT/cwite_cluster_methods`
- `patience`: `15`

### `cwite_global_no_clusters`
- `cluster_control`: `global`
- `config`: `RUN_ROOT/cwite/proposed_best_config.json`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `out_dir`: `RUN_ROOT/cwite_global_no_clusters`
- `patience`: `15`

### `cwite_k_sensitivity`
- `config`: `RUN_ROOT/cwite/proposed_best_config.json`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `k_values`: `['2', '3', '5', '8', '10', '15', '20', '30', '40']`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `out_dir`: `RUN_ROOT/cwite_k_sensitivity`
- `patience`: `15`

### `cwite_random_clusters`
- `cluster_control`: `random`
- `config`: `RUN_ROOT/cwite/proposed_best_config.json`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `out_dir`: `RUN_ROOT/cwite_random_clusters`
- `patience`: `15`
- `random_cluster_seed`: `123`

### `final_report_tables`
- `cluster_model`: `CWITE k=40=RUN_ROOT/cwite_k_sensitivity/cwite_k40_y_test_pred.joblib`
- `data_dir`: `DATA_DIR`
- `horizons`: `['1', '2', '3', '7']`
- `model`: `CWITE=RUN_ROOT/cwite/proposed_feature_sweep_best_y_test_pred.joblib`
- `n_boot`: `1000`
- `no_defaults`: `True`
- `out_dir`: `RUN_ROOT/report_tables`

### `propensity_calibration`
- `data_dir`: `DATA_DIR`
- `n_boot`: `1000`
- `out_dir`: `RUN_ROOT/propensity_ranking_ci`

### `train_cwite`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `hidden_size`: `64`
- `k`: `['5', '10', '20']`
- `lambda_loss`: `['0.05', '0.1', '0.25', '0.5', '0.7', '0.8', '0.9', '1.0']`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `methods`: `Proposed`
- `num_layers`: `3`
- `out_dir`: `RUN_ROOT/cwite`
- `patience`: `15`
- `proposed_loss_scaling`: `normalized`
- `select_dims`: `['50', '100', '200']`
- `transforms`: `['standard', 'select']`
- `weight_func`: `linear`

### `train_deephit`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `hidden_size`: `64`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `methods`: `DeepHit`
- `num_layers`: `3`
- `out_dir`: `RUN_ROOT/deephit`
- `patience`: `15`
- `select_dims`: `['50', '100', '200']`
- `transforms`: `['standard', 'select']`

### `train_ipcw`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `hidden_size`: `64`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `methods`: `IPCW`
- `num_layers`: `3`
- `out_dir`: `RUN_ROOT/ipcw`
- `patience`: `15`
- `select_dims`: `['50', '100', '200']`
- `transforms`: `['standard', 'select']`

### `train_powell`
- `data_dir`: `DATA_DIR`
- `gpu`: `GPU`
- `hidden_size`: `64`
- `log_every_epochs`: `5`
- `max_epochs`: `200`
- `methods`: `Powell`
- `num_layers`: `3`
- `out_dir`: `RUN_ROOT/powell`
- `patience`: `15`
- `select_dims`: `['50', '100', '200']`
- `transforms`: `['standard', 'select']`

