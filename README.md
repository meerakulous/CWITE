# CWITE MLHC 2026 Code Release

This repository contains code accompanying the MLHC 2026 paper on CWITE for time-to-event prediction with informative censoring.

## Directory Structure

- `synthetic/`: synthetic labor experiments, baselines, CWITE variants, calibration analyses, and plotting/result notebooks.
- `semi_synthetic/`: semi-synthetic SUPPORT-based experiments, baselines, CWITE variants, clustering analyses, and result-generation notebooks.
- `real_data/`: real labor analysis code, feature-selection sweeps, clustering sensitivity analyses, propensity-stratified evaluation, calibration checks, and clinical overprediction metrics.
- `hyperparameter_recovery/`: helper scripts and recovered tables used to summarize selected hyperparameters from experiment outputs.

## Data Availability

The synthetic experiments can be regenerated from the included code. The semi-synthetic experiments use SUPPORT-derived data preparation artifacts included with the experiment code where available. The real-data experiments use protected clinical data that cannot be redistributed; those scripts are included to document the analysis pipeline and should be run in an approved environment with the required derived input files.

## Running The Code

Install the Python dependencies listed in `requirements.txt`, then point the scripts at local data and output directories:

```bash
export CWITE_DATA_ROOT=/path/to/cwite-data
export CWITE_OUTPUT_ROOT=/path/to/cwite-outputs
```

The synthetic and semi-synthetic scripts read from `CWITE_DATA_ROOT` and write models/predictions under `CWITE_OUTPUT_ROOT`. The expected subdirectories are:

```text
$CWITE_DATA_ROOT/onevar_data
$CWITE_DATA_ROOT/viol4_onevar_data
$CWITE_DATA_ROOT/support50_propbin_data
$CWITE_OUTPUT_ROOT/onevar_models
$CWITE_OUTPUT_ROOT/onevar_test_preds
$CWITE_OUTPUT_ROOT/support50_propbin_models
$CWITE_OUTPUT_ROOT/support50_propbin_test_preds
```

For real-data analyses, use the real-data-specific variables:

```bash
export CWITE_REAL_DATA_DIR=/path/to/approved/real_labor_joblibs
export CWITE_REAL_RUN_ROOT=/path/to/real_data_outputs
cd real_data
bash run_all_realdata_series.sh
```

The protected clinical data used in the real-data experiments cannot be redistributed, so the real-data scripts require approved local derived joblib files. All formerly local machine paths have been replaced by these environment variables or command-line arguments.

## Hyperparameters

Selected hyperparameters and helper scripts for recovering them from output archives are in `hyperparameter_recovery/`. For synthetic experiments, final selected configurations may vary by independently generated replicate; those replicate-level configurations are documented in the released code and associated output files rather than repeated in full in the manuscript.
