# Hyperparameter Recovery: semi_synthetic

Archive: `<archive-directory>/semi_synthetic_2.tar.gz`

## Paper Table Rows

| method | setting | num_layers | hidden_size | batch_size | lr | weight_decay | propensity_C | k | weight_func | lambda_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ipcw | covariate_low_resource | 1 | 32 | 128 | 0.0005 | 0.0001 | 1.0 |  |  |  |
| ipcw | covariate_moderate_resource | 3 | 16 | 128 | 0.0005 | 0.0001 | 1.0 |  |  |  |
| ipcw | covariate_high_resource | 1 | 32 | 128 | 0.0001 | 1e-06 | 1.0 |  |  |  |
| deephit | covariate_low_resource | 1 | 64 | 128 | 0.0005 | 1e-05 |  |  |  |  |
| deephit | covariate_moderate_resource | 3 | 64 | 64 | 0.001 | 0.0001 |  |  |  |  |
| deephit | covariate_high_resource | 1 | 64 | 64 | 0.001 | 1e-05 |  |  |  |  |
| powell | covariate_low_resource | 3 | 16 | 256 | 0.0001 | 1e-06 |  |  |  |  |
| powell | covariate_moderate_resource | 1 | 16 | 128 | 0.0001 | 1e-05 |  |  |  |  |
| powell | covariate_high_resource | 3 | 32 | 256 | 0.0001 | 1e-06 |  |  |  |  |
| oracle | covariate_low_resource | 3 | 16 | 64 | 0.001 | 1e-06 |  |  |  |  |
| oracle | covariate_moderate_resource | 3 | 32 | 64 | 0.0005 | 0.0001 |  |  |  |  |
| oracle | covariate_high_resource | 3 | 32 | 64 | 0.0005 | 1e-06 |  |  |  |  |
| cwite | covariate_low_resource | 1 | 64 | 128 | 0.0005 | 1e-06 | 1.0 | 95 | linear | 0.075 |
| cwite | covariate_moderate_resource | 1 | 32 | 128 | 0.0005 | 1e-05 | 1.0 | 95 | linear | 0.075 |
| cwite | covariate_high_resource | 1 | 16 | 128 | 0.0005 | 1e-06 | 1.0 | 95 | linear | 0.075 |
| Standard Margin | covariate_low_resource | 3 | 16 | 256 | 0.0001 | 1e-06 |  |  |  |  |
| Standard Margin | covariate_moderate_resource | 1 | 16 | 128 | 0.0001 | 1e-05 |  |  |  |  |
| Standard Margin | covariate_high_resource | 3 | 32 | 256 | 0.0001 | 1e-06 |  |  |  |  |

## Recovered Search Metadata

### `semi_synthetic_2/IPCW.py`
- `argparse_defaults`: `{'idxmin': 0, 'idxmax': -1}`
- `early_stopping_patience`: `15`
- `max_epochs`: `500`
- `optimizer`: `Adam`

### `semi_synthetic_2/IPCW_hyp.py`
- `batch_size_lr_grid`: `{ 64: [1e-3, 5e-4], 128: [5e-4, 1e-4], 256: [1e-4], }`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `num_layers_grid`: `[1, 2, 3]`
- `optimizer`: `Adam`
- `propensity_C_grid`: `[1, 1e-2, 1e-4, 1e-6]`
- `random_grid_trials`: `25`
- `seed`: `42`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `semi_synthetic_2/deephit.py`
- `argparse_defaults`: `{'idxmin': 0, 'idxmax': -1}`
- `early_stopping_patience`: `15`
- `max_epochs`: `500`
- `optimizer`: `Adam`

### `semi_synthetic_2/deephit_hyp.py`
- `batch_size_lr_grid`: `{ 64: [1e-3, 5e-4], 128: [5e-4, 1e-4], 256: [1e-4], }`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `num_layers_grid`: `[1, 2, 3]`
- `optimizer`: `Adam`
- `random_grid_trials`: `25`
- `seed`: `42`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `semi_synthetic_2/oracle.py`
- `argparse_defaults`: `{'idxmin': 0, 'idxmax': -1}`
- `early_stopping_patience`: `15`
- `max_epochs`: `500`
- `optimizer`: `Adam`

### `semi_synthetic_2/oracle_hyp.py`
- `batch_size_lr_grid`: `{ 64: [1e-3, 5e-4], 128: [5e-4, 1e-4], 256: [1e-4], }`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `num_layers_grid`: `[1, 2, 3]`
- `optimizer`: `Adam`
- `random_grid_trials`: `25`
- `seed`: `42`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `semi_synthetic_2/powells.py`
- `argparse_defaults`: `{'idxmin': 0, 'idxmax': -1}`
- `early_stopping_patience`: `15`
- `max_epochs`: `500`
- `optimizer`: `Adam`

### `semi_synthetic_2/powells_hyp.py`
- `batch_size_lr_grid`: `{ 64: [1e-3, 5e-4], 128: [5e-4, 1e-4], 256: [1e-4], }`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `num_layers_grid`: `[1, 2, 3]`
- `optimizer`: `Adam`
- `random_grid_trials`: `25`
- `seed`: `42`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

