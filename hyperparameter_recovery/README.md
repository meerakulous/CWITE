# Hyperparameter recovery scripts

These scripts recover paper-relevant hyperparameters directly from gzipped tar
archives without running the original training code. The synthetic archive on
Desktop is named `synthetic_labor2.targ.z`; that extension is unusual, but the
file is still handled as a valid gzipped tar archive.

## Scripts

- `recover_synthetic_hyperparameters.py`: use on the synthetic experiment archive.
- `recover_semisynthetic_hyperparameters.py`: use on the semi-synthetic archive.
- `recover_real_hyperparameters.py`: use on the real-data code or result archive.

Each script writes three files:

- `*_hyperparameters.json`: full structured recovery output.
- `*_hyperparameters.md`: human-readable summary.
- `*_paper_table_rows.csv`: compact rows for a paper appendix table.

## Commands

```bash
cd <repo-root>

python3 outputs/hyperparameter_recovery/recover_synthetic_hyperparameters.py \
  <archive-directory>/synthetic_labor2.targ.z \
  --out-dir outputs/hyperparameter_recovery/synthetic

python3 outputs/hyperparameter_recovery/recover_semisynthetic_hyperparameters.py \
  <archive-directory>/semi_synthetic_2.tar.gz \
  --out-dir outputs/hyperparameter_recovery/semi_synthetic

python3 outputs/hyperparameter_recovery/recover_real_hyperparameters.py \
  <archive-directory>/real_labor_clean.tar.gz \
  --out-dir outputs/hyperparameter_recovery/real

python3 outputs/hyperparameter_recovery/recover_real_hyperparameters.py \
  <archive-directory>/realdata_selected_hyperparams.tar.gz \
  --out-dir outputs/hyperparameter_recovery/real_selected
```

## Notes

- For synthetic archives, the scripts parse the range-specific files such as `IPCW_hyp_0_125.txt`, `deephit_hyp_0_125.txt`, `powells_hyp_0_125.txt`, and `proposed_orig_hyp_0_125.txt`. The complete per-simulation selected rows are written to CSV; the Markdown shows a preview so it stays readable.
- For semi-synthetic archives, the scripts look for legacy selected-output files such as `IPCW.txt`, `deephit.txt`, `powells.txt`, `oracle.txt`, and `proposed.txt`, then label columns like `num_layers`, `hidden_size`, `batch_size`, `lr`, `weight_decay`, `k`, `lambda_loss`, and `weight_func`.
- For real-data archives, selected configs are recovered when the archive contains `*_best_config.json` or `sweep_results.csv`. The code-only `real_labor_clean.tar.gz` recovers the command-level search space; the result archive `realdata_selected_hyperparams.tar.gz` recovers the exact selected configs used in the paper.
