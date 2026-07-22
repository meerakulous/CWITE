# Hyperparameter Recovery: real

Archive: `/Users/meerakrishnamoorthy/Desktop/realdata_selected_hyperparams.tar.gz`

## Paper Table Rows

| method | setting | transform | select_dim | num_layers | hidden_size | batch_size | lr | weight_decay | propensity_C | k | weight_func | lambda_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IPCW | real_data_selected | select | 100 | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| Standard | real_data_selected | select | 200 | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| Standard Margin | real_data_selected | select | 200 | 3 | 64 | 128 | 0.0001 | 0.0001 |  |  |  |  |
| CWITE | real_data_selected | select | 200 | 3 | 64 | 128 | 0.0001 | 0.0001 |  | 5 | linear | 1.0 |

## Recovered Search Metadata

## Selected JSON Configs

### `data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json`
- `batch_size`: `128`
- `best_epoch`: `29`
- `best_val_loss`: `4.4567444523175554`
- `dim`: `200`
- `error`: ``
- `hidden_size`: `64`
- `k`: `5`
- `lambda_loss`: `1.0`
- `lr`: `0.0001`
- `method`: `Proposed`
- `num_layers`: `3`
- `proposed_loss_scaling`: `normalized`
- `seconds`: `29.068`
- `seed`: `42`
- `status`: `ok`
- `transform`: `select`
- `val_high_mae`: `5.573394495412844`
- `val_low_mae`: `6.9393939393939394`
- `val_overall_mae`: `6.0130456598093325`
- `val_score`: `6.0130456598093325`
- `wd`: `0.0001`
- `weight_func`: `linear`

### `data4/meerak/cwite_realdata_final/deephit/deephit_best_config.json`
- `batch_size`: `128`
- `best_epoch`: `17`
- `best_val_loss`: `1.8525804943508573`
- `dim`: `200`
- `error`: ``
- `hidden_size`: `64`
- `k`: `0`
- `lambda_loss`: `0.0`
- `lr`: `0.0001`
- `method`: `DeepHit`
- `num_layers`: `3`
- `proposed_loss_scaling`: `normalized`
- `seconds`: `23.65`
- `seed`: `42`
- `status`: `ok`
- `transform`: `select`
- `val_high_mae`: `7.298165137614679`
- `val_low_mae`: `11.727272727272727`
- `val_overall_mae`: `7.915203211239338`
- `val_score`: `7.915203211239338`
- `wd`: `0.0001`
- `weight_func`: `none`

### `data4/meerak/cwite_realdata_final/ipcw/ipcw_best_config.json`
- `batch_size`: `128`
- `best_epoch`: `26`
- `best_val_loss`: `3.5791246559884815`
- `dim`: `100`
- `error`: ``
- `hidden_size`: `64`
- `k`: `0`
- `lambda_loss`: `0.0`
- `lr`: `0.0001`
- `method`: `IPCW`
- `num_layers`: `3`
- `proposed_loss_scaling`: `normalized`
- `seconds`: `24.65`
- `seed`: `42`
- `status`: `ok`
- `transform`: `select`
- `val_high_mae`: `5.637614678899083`
- `val_low_mae`: `7.454545454545454`
- `val_overall_mae`: `6.159558454591068`
- `val_score`: `6.159558454591068`
- `wd`: `0.0001`
- `weight_func`: `none`

### `data4/meerak/cwite_realdata_final/powell/powell_best_config.json`
- `batch_size`: `128`
- `best_epoch`: `48`
- `best_val_loss`: `2.2483364045619965`
- `dim`: `200`
- `error`: ``
- `hidden_size`: `64`
- `k`: `0`
- `lambda_loss`: `0.0`
- `lr`: `0.0001`
- `method`: `Powell`
- `num_layers`: `3`
- `proposed_loss_scaling`: `normalized`
- `seconds`: `36.986`
- `seed`: `42`
- `status`: `ok`
- `transform`: `select`
- `val_high_mae`: `6.7110091743119265`
- `val_low_mae`: `7.696969696969697`
- `val_overall_mae`: `7.0070245860511795`
- `val_score`: `7.0070245860511795`
- `wd`: `0.0001`
- `weight_func`: `none`

