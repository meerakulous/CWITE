# Hyperparameter Recovery: real

Archive: `/Users/meerakrishnamoorthy/Desktop/real_labor_clean.tar.gz`

## Paper Table Rows

| method | setting | num_layers | hidden_size | batch_size | lr | weight_decay | propensity_C | k | weight_func | lambda_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IPCW | real_data_feature_sweep | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| DeepHit | real_data_feature_sweep | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| Standard Margin | real_data_feature_sweep | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| CWITE | real_data_feature_sweep | 3 | 64 | 128 | 0.0001 | 0.0001 |  | 5; 10; 20 | linear | 0.05; 0.1; 0.25; 0.5; 0.7; 0.8; 0.9; 1.0 |

## Recovered Search Metadata

### `real_labor_clean/proposed_cluster_method_controls.py`
- `argparse_defaults`: `{'data_dir': '/data4/meerak/real_labor', 'out_dir': '/data4/meerak/proposed_cluster_method_controls', 'cluster_methods': '["kmeans"', 'cluster_pca_dim': 20, 'random_cluster_seed': 123, 'max_epochs': 200, 'patience': 15, 'log_every_epochs': 5, 'low_max': 0.1, 'high_min': 0.7, 'high_penalty': 0.25, 'prediction_mode': 'argmax', 'selection_score': 'overall', 'log_file': None}`

### `real_labor_clean/proposed_k_sensitivity.py`
- `argparse_defaults`: `{'data_dir': '/data4/meerak/real_labor', 'out_dir': '/data4/meerak/cwite_realdata_final/cwite_k_sensitivity', 'k_values': '[2', 'max_epochs': 200, 'patience': 15, 'log_every_epochs': 5, 'low_max': 0.1, 'high_min': 0.7, 'high_penalty': 0.25, 'prediction_mode': 'argmax', 'selection_score': 'overall', 'log_file': None}`

### `real_labor_clean/proposed_random_cluster_control.py`
- `argparse_defaults`: `{'data_dir': '/data4/meerak/real_labor', 'out_dir': '/data4/meerak/proposed_random_cluster_control', 'cluster_control': 'random', 'random_cluster_seed': 123, 'max_epochs': None, 'patience': None, 'log_every_epochs': 5, 'low_max': 0.1, 'high_min': 0.7, 'high_penalty': 0.25, 'prediction_mode': 'argmax', 'selection_score': 'overall', 'log_file': None}`

### `real_labor_clean/realdata_overall_sweep.py`
- `argparse_defaults`: `{'data_dir': '/data4/meerak/real_labor', 'out_dir': '/data4/meerak/cwite_realdata_final', 'methods': 'METHODS', 'seed': 42, 'max_epochs': 300, 'patience': 20, 'low_max': 0.1, 'high_min': 0.7, 'high_penalty': 0.25, 'k': '[10', 'weight_func': '["linear"', 'lambda_loss': '[0.05', 'transforms': '["standard"', 'pca_dims': '[10', 'select_dims': '[50', 'num_layers': 2, 'hidden_size': 32, 'batch_size': 128, 'lr': 0.0001, 'wd': 0.0001, 'prediction_mode': 'argmax', 'proposed_loss_scaling': 'normalized', 'selection_score': 'overall', 'log_every_epochs': 5, 'log_file': None, 'max_configs': None}`
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

