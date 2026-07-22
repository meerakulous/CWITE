# Real-data CWITE experiments

This folder contains the cleaned-up scripts for reproducing the real-data experiments and
reviewer-response analyses. The exploratory scripts in the parent `retrain/` folder are
not needed for the collaborator-facing run.

## Files

- `realdata_overall_sweep.py`
  - Main real-data training/sweep script.
  - Uses argmax TTE predictions.
  - Uses old-script-style model selection: best epoch by validation loss, best config by
    overall uncensored validation MAE.
  - Supports shared preprocessing for all methods: all standardized features or
    supervised feature-selected subsets.
- `check_feature_sweep_leaderboard.py`
  - Reads running sweep folders and prints current validation leaderboard.
- `generate_realdata_report_tables.py`
  - Generates final MAE, c-index, clinical overprediction, and cluster-sensitivity tables
    from saved prediction files.
- `propensity_ranking_ci.py`
  - Computes propensity-model AUROC CI, Brier CI, Spearman ranking CI, and observed
    uncensoring rate by propensity bin.
- `evaluate_clinical_decision_metrics.py`
  - Computes overprediction metrics from saved prediction files.
  - This is useful if you only want the clinical/overprediction table.
- `proposed_random_cluster_control.py`
  - Retrains the selected CWITE config once with random clusters or with one global
    no-clustering reference.
- `proposed_cluster_method_controls.py`
  - Retrains the selected CWITE config once per clustering method: k-means,
    k-means+PCA, GMM, spectral clustering, random, or global.
- `proposed_k_sensitivity.py`
  - Retrains the selected CWITE config once per requested number of k-means clusters.

## Expected data layout

The scripts expect joblib files named:

```text
real_labor_X_train.joblib
real_labor_y_train.joblib
real_labor_binary_y_train.joblib
real_labor_X_val.joblib
real_labor_y_val.joblib
real_labor_binary_y_val.joblib
real_labor_X_test.joblib
real_labor_y_test.joblib
real_labor_binary_y_test.joblib
```

Default server data directory:

```bash
/data4/meerak/real_labor
```

## 1. One-command serial run

For a clean collaborator run, use the serial runner. It writes everything under one root
folder with clean names and never emits stale legacy CWITE rows.

Default output root:

```bash
/data4/meerak/cwite_realdata_final
```

Run:

```bash
source ~/venv/bin/activate
cd /data/home/meerak/real_labor_updated/real_data_cleaned_up_experiments

DATA_DIR=/data4/meerak/real_labor \
RUN_ROOT=/data4/meerak/cwite_realdata_final \
GPU=0 \
bash run_all_realdata_series.sh
```

The runner executes, in order:

1. IPCW
2. DeepHit
3. Powell
4. CWITE
5. CWITE random-cluster control
6. CWITE global/no-cluster control
7. CWITE clustering-method sensitivity
8. CWITE k sensitivity
9. Propensity calibration/ranking CIs
10. Final MAE, c-index, overprediction, and clustering sensitivity tables

Final outputs:

```text
/data4/meerak/cwite_realdata_final/report_tables/all_report_tables.tex
/data4/meerak/cwite_realdata_final/propensity_ranking_ci/propensity_bin_ranking_table.tex
/data4/meerak/cwite_realdata_final/logs/
```

Clean prediction/config paths:

```text
/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json
```

## 2. Optional: train the four main methods in parallel

Use separate `screen` sessions so each method has its own GPU and output folder.

### IPCW

```bash
screen -S realdata_ipcw
source ~/venv/bin/activate
cd /data/home/meerak/real_labor_updated

python realdata_overall_sweep.py \
  -gpu 0 \
  --methods IPCW \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/ipcw \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5 \
  --transforms standard select \
  --select-dims 50 100 200 \
  --num-layers 3 \
  --hidden-size 64
```

Detach with `Ctrl-a d`.

### DeepHit

```bash
screen -S realdata_deephit
source ~/venv/bin/activate
cd /data/home/meerak/real_labor_updated

python realdata_overall_sweep.py \
  -gpu 1 \
  --methods DeepHit \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/deephit \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5 \
  --transforms standard select \
  --select-dims 50 100 200 \
  --num-layers 3 \
  --hidden-size 64
```

### Powell

```bash
screen -S realdata_powell
source ~/venv/bin/activate
cd /data/home/meerak/real_labor_updated

python realdata_overall_sweep.py \
  -gpu 2 \
  --methods Powell \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/powell \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5 \
  --transforms standard select \
  --select-dims 50 100 200 \
  --num-layers 3 \
  --hidden-size 64
```

### CWITE

```bash
screen -S realdata_cwite
source ~/venv/bin/activate
cd /data/home/meerak/real_labor_updated

python realdata_overall_sweep.py \
  -gpu 4 \
  --methods Proposed \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/cwite \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5 \
  --transforms standard select \
  --select-dims 50 100 200 \
  --num-layers 3 \
  --hidden-size 64 \
  --k 5 10 20 \
  --weight-func linear \
  --lambda-loss 0.05 0.1 0.25 0.5 0.7 0.8 0.9 1.0 \
  --proposed-loss-scaling normalized
```

## 3. Monitor training

```bash
python check_feature_sweep_leaderboard.py \
  "IPCW overall=/data4/meerak/cwite_realdata_final/ipcw" \
  "DeepHit overall=/data4/meerak/cwite_realdata_final/deephit" \
  "Powell overall=/data4/meerak/cwite_realdata_final/powell" \
  "CWITE overall=/data4/meerak/cwite_realdata_final/cwite"
```

The main saved prediction files are:

```text
/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_pred.joblib
```

Best configs are saved beside the prediction files as `*_best_config.json`.

## 4. Generate final MAE, c-index, and overprediction tables

This command uses the default output paths listed above.

```bash
python generate_realdata_report_tables.py \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/final_realdata_report_tables \
  --n-boot 1000 \
  --horizons 1 2 3 7
```

Outputs:

```text
/data4/meerak/final_realdata_report_tables/main_extreme_bin_tables.tex
/data4/meerak/final_realdata_report_tables/clinical_overprediction_tables.tex
/data4/meerak/final_realdata_report_tables/cluster_sensitivity_tables.tex
/data4/meerak/final_realdata_report_tables/all_report_tables.tex
```

If cluster-sensitivity files are not available yet, pass explicit `--model` arguments or
run the clinical-only command in Section 5.

## 5. Propensity-model discrimination and calibration

Run:

```bash
python propensity_ranking_ci.py \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/propensity_ranking_ci \
  --n-boot 1000
```

Outputs:

```text
/data4/meerak/propensity_ranking_ci/propensity_summary_ci.csv
/data4/meerak/propensity_ranking_ci/propensity_bin_ranking_ci.csv
/data4/meerak/propensity_ranking_ci/propensity_bin_ranking_table.tex
```

This reports:

- Test AUROC with 95% bootstrap CI.
- Test Brier score with 95% bootstrap CI.
- Spearman correlation between predicted propensity and uncensoring indicator.
- Observed uncensoring rate with 95% Wilson CI in each propensity bin.

## 6. Clinical overprediction metrics only

Use this when you only need the overprediction rate table.

```bash
python evaluate_clinical_decision_metrics.py \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/clinical_decision_metrics \
  --model "IPCW=/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_pred.joblib" \
  --model "DeepHit=/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_pred.joblib" \
  --model "Powell=/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_pred.joblib" \
  --model "CWITE=/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_pred.joblib" \
  --horizons 1 2 3 7 \
  --n-boot 1000
```

Key clinical metrics:

- `% predicted after true TTE`
- `Mean days predicted after true TTE`
- `% overpredicted by at least 1d`
- `% overpredicted by at least 7d`

## 7. Clustering sensitivity analyses

These scripts start from the selected CWITE config:

```text
/data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json
```

They do not tune CWITE again. They hold the selected architecture, feature representation,
loss weight, weighting function, and `k` fixed, and retrain while changing only the cluster
assignment procedure.

### Random clusters and no-clustering global reference

Random clusters:

```bash
python proposed_random_cluster_control.py \
  -gpu 5 \
  --config /data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/cwite_random_clusters \
  --cluster-control random \
  --random-cluster-seed 123 \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5
```

No-clustering/global-reference control:

```bash
python proposed_random_cluster_control.py \
  -gpu 5 \
  --config /data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/cwite_global_no_clusters \
  --cluster-control global \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5
```

Outputs:

```text
/data4/meerak/cwite_realdata_final/cwite_random_clusters/random_cluster_proposed_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite_global_no_clusters/global_cluster_proposed_y_test_pred.joblib
```

### Alternative clustering methods

```bash
python proposed_cluster_method_controls.py \
  -gpu 5 \
  --config /data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/cwite_cluster_methods \
  --cluster-methods kmeans kmeans_pca gmm spectral \
  --cluster-pca-dim 20 \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5
```

Outputs:

```text
/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_kmeans_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_kmeans_pca_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_gmm_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite_cluster_methods/cwite_spectral_y_test_pred.joblib
```

After these files exist, rerun `generate_realdata_report_tables.py` to create the
cluster-sensitivity table.

### D-calibration

D-calibration requires full predictive distributions, not just the final point
predictions used for MAE. To save test-set PMFs during a sweep, add
`--save-test-distribution` to the training commands. This writes files like:

```text
/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_dist.joblib
```

Then run:

```bash
python evaluate_d_calibration.py \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/d_calibration \
  --model "IPCW=/data4/meerak/cwite_realdata_final/ipcw/ipcw_feature_sweep_best_y_test_dist.joblib" \
  --model "DeepHit=/data4/meerak/cwite_realdata_final/deephit/deephit_feature_sweep_best_y_test_dist.joblib" \
  --model "Powell=/data4/meerak/cwite_realdata_final/powell/powell_feature_sweep_best_y_test_dist.joblib" \
  --model "CWITE=/data4/meerak/cwite_realdata_final/cwite/proposed_feature_sweep_best_y_test_dist.joblib"
```

The script computes uncensored D-calibration by testing whether
`\hat F(T_i)` is uniform among uncensored test individuals. Point-prediction
files are intentionally skipped unless `--allow-point-mass` is passed; that
mode is diagnostic only and should not be reported as survival D-calibration.

### Number-of-clusters sensitivity

This holds the selected CWITE config fixed and retrains once for each requested `k`.

```bash
python proposed_k_sensitivity.py \
  -gpu 5 \
  --config /data4/meerak/cwite_realdata_final/cwite/proposed_best_config.json \
  --data-dir /data4/meerak/real_labor \
  --out-dir /data4/meerak/cwite_realdata_final/cwite_k_sensitivity \
  --k-values 2 3 5 8 10 15 20 30 40 \
  --max-epochs 200 \
  --patience 15 \
  --log-every-epochs 5
```

Outputs:

```text
/data4/meerak/cwite_realdata_final/cwite_k_sensitivity/cwite_k5_y_test_pred.joblib
/data4/meerak/cwite_realdata_final/cwite_k_sensitivity/cwite_k_sensitivity_results.csv
```

## Notes for collaborators

- Do not use legacy CWITE rows from older outputs. The final CWITE prediction
  file is `cwite/proposed_feature_sweep_best_y_test_pred.joblib`.
- The final main analysis uses argmax predictions.
- The final selection rule is overall uncensored validation MAE, with best epoch selected
  by validation loss.
- Feature selection is fit on training only, using uncensored training TTEs.
- The shared preprocessing grid lets every method choose between all standardized features
  and selected feature subsets, so feature reduction is not applied only to CWITE.
