#!/bin/bash
# SLURM batch submission script for EMA vs Weight Averaging sweep
# Usage: ./run_method_sweep_slurm.sh [experiment_name]
#
# Runs ema_holdout experiment with math500 selection and gsm8k validation,
# comparing EMA (beta=0.5) to weight averaging.

EXPERIMENT_NAME=${1:-"ema_holdout"}
SELECTION_DATASETS='[math500]'
VALIDATION_DATASETS='[gsm8k]'

echo "Starting method sweep for experiment: ${EXPERIMENT_NAME}"
echo "Methods: EMA (beta=0.5), Weight Averaging"
echo "Selection datasets: ${SELECTION_DATASETS}"
echo "Validation datasets: ${VALIDATION_DATASETS}"
echo ""

JOB_IDS=()

# EMA run
OUTPUT_DIR="outputs/method_sweep/${EXPERIMENT_NAME}_ema"
echo "Submitting job for EMA"
JOB_OUTPUT=$(sbatch \
    --job-name="ema" \
    slurm/run_ferranti_multi_gpu.sh \
    experiment=${EXPERIMENT_NAME} \
    merge.method=ema \
    merge.ema.beta=0.5 \
    selection_datasets=${SELECTION_DATASETS} \
    validation_datasets=${VALIDATION_DATASETS} \
    output_dir=${OUTPUT_DIR})
JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')
JOB_IDS+=($JOB_ID)
echo "  Job ID: ${JOB_ID}"
echo ""

# Weight Averaging run
OUTPUT_DIR="outputs/method_sweep/${EXPERIMENT_NAME}_weight_averaging"
echo "Submitting job for Weight Averaging"
JOB_OUTPUT=$(sbatch \
    --job-name="wa" \
    slurm/run_ferranti_multi_gpu.sh \
    experiment=${EXPERIMENT_NAME} \
    merge.method=weight_averaging_running \
    selection_datasets=${SELECTION_DATASETS} \
    validation_datasets=${VALIDATION_DATASETS} \
    output_dir=${OUTPUT_DIR})
JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')
JOB_IDS+=($JOB_ID)
echo "  Job ID: ${JOB_ID}"
echo ""

echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check specific job: squeue -j <JOB_ID>"
echo "Cancel all: scancel ${JOB_IDS[@]}"
echo ""
echo "After completion, analyze results with:"
echo "  scripts/analyze_method_sweep.sh ${EXPERIMENT_NAME}"
echo ""
echo "To visualize:"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME} --base-dir outputs/method_sweep"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME} --base-dir outputs/method_sweep --selection"
