# CWITE MLHC 2026 Code Release

This repository contains code accompanying the MLHC 2026 paper on CWITE for time-to-event prediction with informative censoring.

## Directory Structure

- `synthetic/`: synthetic labor experiments, baselines, CWITE variants, calibration analyses, and plotting/result notebooks.
- `semi_synthetic/`: semi-synthetic SUPPORT-based experiments, baselines, CWITE variants, clustering analyses, and result-generation notebooks.
- `real_data/`: real labor analysis code, feature-selection sweeps, clustering sensitivity analyses, propensity-stratified evaluation, calibration checks, and clinical overprediction metrics.
- `hyperparameter_recovery/`: helper scripts and recovered tables used to summarize selected hyperparameters from experiment outputs.

## Data Availability

The synthetic experiments can be regenerated from the included code. The semi-synthetic experiments use SUPPORT-derived data preparation artifacts included with the experiment code where available. The real-data experiments use protected clinical data that cannot be redistributed; those scripts are included to document the analysis pipeline and should be run in an approved environment with the required derived input files.

## Notes For Running

Many scripts were originally run on the authors' compute environment and still contain absolute paths such as `/data4/meerak/...`. Before rerunning the experiments in a new environment, update those paths or mount the expected directories.

The main Python dependencies are listed in `requirements.txt`. Exact package versions may depend on the target compute environment.

## Hyperparameters

Selected hyperparameters and helper scripts for recovering them from output archives are in `hyperparameter_recovery/`. For synthetic experiments, final selected configurations may vary by independently generated replicate; those replicate-level configurations are documented in the released code and associated output files rather than repeated in full in the manuscript.
