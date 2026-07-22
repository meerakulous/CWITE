# Hyperparameter Recovery: synthetic

Archive: `/Users/meerakrishnamoorthy/Desktop/synthetic_labor2.targ.z`

## Paper Table Rows

5914 rows were recovered. The complete table is in the companion CSV; this preview shows the first 40 rows.

| method | setting | num_layers | hidden_size | batch_size | lr | weight_decay | propensity_C | k | weight_func | lambda_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| powell | COUNT0-125 | 3 | 32 | 512 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 256 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.001 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 256 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 128 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 128 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 128 | 0.001 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 512 | 0.001 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 256 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 1024 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 512 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 128 | 0.001 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 1024 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 256 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 256 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 1024 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 1024 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 1024 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 512 | 0.001 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 128 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.001 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 1024 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 128 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 32 | 1024 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 512 | 0.01 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.001 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 16 | 512 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.001 | 1e-06 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 128 | 0.01 | 0.0001 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 256 | 0.01 | 1e-05 |  |  |  |  |
| powell | COUNT0-125 | 3 | 64 | 256 | 0.001 | 1e-06 |  |  |  |  |

## Recovered Search Metadata

### `synthetic_labor2/IPCW.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `propensity_C_grid`: `[1, 1e-2, 1e-4, 1e-6]`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/d_calibration/IPCW.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `propensity_C_grid`: `[1, 1e-2, 1e-4, 1e-6]`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/d_calibration/deephit.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/d_calibration/oracle.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[32, 64]`
- `max_epochs`: `10000`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-5, 1e-6]`

### `synthetic_labor2/d_calibration/powells.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/d_calibration/proposed.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `lambda_loss_grid`: `[1e-2, 0.05, 1e-1, 0.5, 1]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `propensity_C_grid`: `[1, 1e-2, 1e-4, 1e-6]`
- `random_grid_trials`: `10`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`
- `weight_func_grid`: `['linear', 'exp', 'poly']`

### `synthetic_labor2/deephit.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/oracle.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[32, 64]`
- `max_epochs`: `10000`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-5, 1e-6]`

### `synthetic_labor2/powells.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `random_grid_trials`: `5`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`

### `synthetic_labor2/proposed.py`
- `early_stopping_patience`: `15`
- `hidden_size_grid`: `[16, 32, 64]`
- `lambda_loss_grid`: `[1e-2, 0.05, 1e-1, 0.5, 1]`
- `max_epochs`: `500`
- `optimizer`: `Adam`
- `propensity_C_grid`: `[1, 1e-2, 1e-4, 1e-6]`
- `random_grid_trials`: `10`
- `weight_decay_grid`: `[1e-4, 1e-5, 1e-6]`
- `weight_func_grid`: `['linear', 'exp', 'poly']`

## Synthetic Row Counts

- `cwite`: 1000 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 125, COUNT750-875: 125, COUNT875-1000: 125)
- `deephit`: 1000 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 125, COUNT750-875: 125, COUNT875-1000: 125)
- `ipcw`: 1000 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 125, COUNT750-875: 125, COUNT875-1000: 125)
- `oracle`: 914 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 39, COUNT750-875: 125, COUNT875-1000: 125)
- `powell`: 1000 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 125, COUNT750-875: 125, COUNT875-1000: 125)
- `standard_margin`: 1000 rows (COUNT0-125: 125, COUNT125-250: 125, COUNT250-375: 125, COUNT375-500: 125, COUNT500-625: 125, COUNT625-750: 125, COUNT750-875: 125, COUNT875-1000: 125)

