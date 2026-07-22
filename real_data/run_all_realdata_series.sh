#!/usr/bin/env bash
set -euo pipefail

# Serial end-to-end real-data run for collaborator handoff.
# Override these when launching if needed:
#   DATA_DIR=/path/to/data RUN_ROOT=/path/to/output GPU=0 bash run_all_realdata_series.sh

DATA_DIR="${DATA_DIR:-${CWITE_REAL_DATA_DIR:-../data/real_labor}}"
RUN_ROOT="${RUN_ROOT:-${CWITE_REAL_RUN_ROOT:-../outputs/real_data}}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RUN_ROOT}/logs"

cd "${SCRIPT_DIR}"

echo "Run root: ${RUN_ROOT}"
echo "Data dir:  ${DATA_DIR}"
echo "GPU:       ${GPU}"
echo "Python:    ${PYTHON}"

run_and_log() {
  local name="$1"
  shift
  echo
  echo "================================================================================"
  echo "START ${name}: $(date)"
  echo "LOG ${RUN_ROOT}/logs/${name}.log"
  echo "================================================================================"
  "$@" 2>&1 | tee "${RUN_ROOT}/logs/${name}.log"
  echo "DONE ${name}: $(date)"
}

run_and_log train_ipcw \
  "${PYTHON}" realdata_overall_sweep.py \
    -gpu "${GPU}" \
    --methods IPCW \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/ipcw" \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5 \
    --transforms standard select \
    --select-dims 50 100 200 \
    --num-layers 3 \
    --hidden-size 64

run_and_log train_deephit \
  "${PYTHON}" realdata_overall_sweep.py \
    -gpu "${GPU}" \
    --methods DeepHit \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/deephit" \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5 \
    --transforms standard select \
    --select-dims 50 100 200 \
    --num-layers 3 \
    --hidden-size 64

run_and_log train_powell \
  "${PYTHON}" realdata_overall_sweep.py \
    -gpu "${GPU}" \
    --methods Powell \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/powell" \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5 \
    --transforms standard select \
    --select-dims 50 100 200 \
    --num-layers 3 \
    --hidden-size 64

run_and_log train_cwite \
  "${PYTHON}" realdata_overall_sweep.py \
    -gpu "${GPU}" \
    --methods Proposed \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/cwite" \
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

run_and_log cwite_random_clusters \
  "${PYTHON}" proposed_random_cluster_control.py \
    -gpu "${GPU}" \
    --config "${RUN_ROOT}/cwite/proposed_best_config.json" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/cwite_random_clusters" \
    --cluster-control random \
    --random-cluster-seed 123 \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5

run_and_log cwite_global_no_clusters \
  "${PYTHON}" proposed_random_cluster_control.py \
    -gpu "${GPU}" \
    --config "${RUN_ROOT}/cwite/proposed_best_config.json" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/cwite_global_no_clusters" \
    --cluster-control global \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5

run_and_log cwite_cluster_methods \
  "${PYTHON}" proposed_cluster_method_controls.py \
    -gpu "${GPU}" \
    --config "${RUN_ROOT}/cwite/proposed_best_config.json" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/cwite_cluster_methods" \
    --cluster-methods kmeans kmeans_pca gmm spectral \
    --cluster-pca-dim 20 \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5

run_and_log cwite_k_sensitivity \
  "${PYTHON}" proposed_k_sensitivity.py \
    -gpu "${GPU}" \
    --config "${RUN_ROOT}/cwite/proposed_best_config.json" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/cwite_k_sensitivity" \
    --k-values 2 3 5 8 10 15 20 30 40 \
    --max-epochs 200 \
    --patience 15 \
    --log-every-epochs 5

run_and_log propensity_calibration \
  "${PYTHON}" propensity_ranking_ci.py \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/propensity_ranking_ci" \
    --n-boot 1000

run_and_log final_report_tables \
  "${PYTHON}" generate_realdata_report_tables.py \
    --data-dir "${DATA_DIR}" \
    --out-dir "${RUN_ROOT}/report_tables" \
    --no-defaults \
    --model "IPCW=${RUN_ROOT}/ipcw/ipcw_feature_sweep_best_y_test_pred.joblib" \
    --model "DeepHit=${RUN_ROOT}/deephit/deephit_feature_sweep_best_y_test_pred.joblib" \
    --model "Powell=${RUN_ROOT}/powell/powell_feature_sweep_best_y_test_pred.joblib" \
    --model "CWITE=${RUN_ROOT}/cwite/proposed_feature_sweep_best_y_test_pred.joblib" \
    --cluster-model "CWITE KMeans=${RUN_ROOT}/cwite/proposed_feature_sweep_best_y_test_pred.joblib" \
    --cluster-model "CWITE Random Clusters=${RUN_ROOT}/cwite_random_clusters/random_cluster_proposed_y_test_pred.joblib" \
    --cluster-model "CWITE Global No Clusters=${RUN_ROOT}/cwite_global_no_clusters/global_cluster_proposed_y_test_pred.joblib" \
    --cluster-model "CWITE KMeans rerun=${RUN_ROOT}/cwite_cluster_methods/cwite_kmeans_y_test_pred.joblib" \
    --cluster-model "CWITE KMeans+PCA=${RUN_ROOT}/cwite_cluster_methods/cwite_kmeans_pca_y_test_pred.joblib" \
    --cluster-model "CWITE GMM=${RUN_ROOT}/cwite_cluster_methods/cwite_gmm_y_test_pred.joblib" \
    --cluster-model "CWITE Spectral=${RUN_ROOT}/cwite_cluster_methods/cwite_spectral_y_test_pred.joblib" \
    --cluster-model "CWITE k=2=${RUN_ROOT}/cwite_k_sensitivity/cwite_k2_y_test_pred.joblib" \
    --cluster-model "CWITE k=3=${RUN_ROOT}/cwite_k_sensitivity/cwite_k3_y_test_pred.joblib" \
    --cluster-model "CWITE k=5=${RUN_ROOT}/cwite_k_sensitivity/cwite_k5_y_test_pred.joblib" \
    --cluster-model "CWITE k=8=${RUN_ROOT}/cwite_k_sensitivity/cwite_k8_y_test_pred.joblib" \
    --cluster-model "CWITE k=10=${RUN_ROOT}/cwite_k_sensitivity/cwite_k10_y_test_pred.joblib" \
    --cluster-model "CWITE k=15=${RUN_ROOT}/cwite_k_sensitivity/cwite_k15_y_test_pred.joblib" \
    --cluster-model "CWITE k=20=${RUN_ROOT}/cwite_k_sensitivity/cwite_k20_y_test_pred.joblib" \
    --cluster-model "CWITE k=30=${RUN_ROOT}/cwite_k_sensitivity/cwite_k30_y_test_pred.joblib" \
    --cluster-model "CWITE k=40=${RUN_ROOT}/cwite_k_sensitivity/cwite_k40_y_test_pred.joblib" \
    --n-boot 1000 \
    --horizons 1 2 3 7

echo
echo "All done."
echo "Main report:        ${RUN_ROOT}/report_tables/all_report_tables.tex"
echo "Propensity report:  ${RUN_ROOT}/propensity_ranking_ci/propensity_bin_ranking_table.tex"
echo "Logs:               ${RUN_ROOT}/logs"
