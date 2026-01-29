#!/bin/bash
# SLURM batch submission script for beta value sweep
# Usage: ./run_beta_sweep_slurm.sh [experiment_name]

EXPERIMENT_NAME=${1:-"ema_holdout"}
BETA_VALUES=(0.1 0.3 0.5 0.7 0.9)

# Selection and validation datasets
SELECTION_DATASETS='[gsm8k]'
VALIDATION_DATASETS='[math500]'

echo "Starting beta sweep for experiment: ${EXPERIMENT_NAME}"
echo "Beta values: ${BETA_VALUES[@]}"
echo "Selection datasets: ${SELECTION_DATASETS}"
echo "Validation datasets: ${VALIDATION_DATASETS}"
echo ""

# Submit jobs for each beta value
JOB_IDS=()
for BETA in "${BETA_VALUES[@]}"; do
    OUTPUT_DIR="outputs/beta_sweep/${EXPERIMENT_NAME}_beta_${BETA}"

    echo "Submitting job for beta=${BETA}"
    JOB_OUTPUT=$(sbatch \
        --job-name="beta_${BETA}" \
        slurm/run_ferranti_multi_gpu.sh \
        experiment=${EXPERIMENT_NAME} \
        merge.ema.beta=${BETA} \
        selection_datasets=${SELECTION_DATASETS} \
        validation_datasets=${VALIDATION_DATASETS} \
        output_dir=${OUTPUT_DIR})

    # Extract job ID from sbatch output
    JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: ${JOB_ID}"
    echo ""
done

echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check specific job: squeue -j <JOB_ID>"
echo "Cancel all: scancel ${JOB_IDS[@]}"
echo ""
echo "After completion, analyze results with:"
echo "  scripts/analyze_beta_sweep.sh ${EXPERIMENT_NAME}"
