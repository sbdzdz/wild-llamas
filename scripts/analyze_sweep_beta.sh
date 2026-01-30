#!/bin/bash
# Analysis script for beta sweep results
# Usage: ./analyze_beta_sweep.sh [experiment_name]

EXPERIMENT_NAME=${1:-"ema_holdout"}
BASE_DIR="outputs/beta_sweep"

echo "Analyzing beta sweep results for: ${EXPERIMENT_NAME}"
echo "========================================="
echo ""

# Check if results directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo "Error: Results directory ${BASE_DIR} not found"
    exit 1
fi

# Find all beta experiment directories
BETA_DIRS=$(find ${BASE_DIR} -maxdepth 1 -type d -name "${EXPERIMENT_NAME}_beta_*" | sort)

if [ -z "${BETA_DIRS}" ]; then
    echo "Error: No results found for experiment ${EXPERIMENT_NAME}"
    echo "Expected directories matching: ${BASE_DIR}/${EXPERIMENT_NAME}_beta_*"
    exit 1
fi

echo "Found results for the following beta values:"
for DIR in ${BETA_DIRS}; do
    BETA=$(basename ${DIR} | sed -n 's/.*beta_\([0-9.]*\).*/\1/p')
    echo "  - beta=${BETA}: ${DIR}"
done
echo ""

# Extract results from each run
echo ""
echo "Results Summary:"
echo "----------------"
printf "%-8s %-20s %-20s %-18s %-12s\n" "Beta" "Selection Acc" "Validation Acc" "Models Merged" "Final Step"
printf "%-8s %-20s %-20s %-18s %-12s\n" "----" "--------------" "---------------" "-------------" "----------"

for DIR in ${BETA_DIRS}; do
    BETA=$(basename ${DIR} | sed -n 's/.*beta_\([0-9.]*\).*/\1/p')
    MERGE_LOG="${DIR}/merge_log.csv"

    if [ ! -f "${MERGE_LOG}" ]; then
        echo "Warning: No merge_log.csv found in ${DIR}"
        continue
    fi

    # Get the last row (final merged model performance)
    LAST_ROW=$(tail -n 1 ${MERGE_LOG})

    # Extract metrics (columns: model_id, current_accuracy, merged_accuracy_partial, merged_accuracy_full, validation_accuracy, num_eval_samples)
    SELECTION_ACC=$(echo ${LAST_ROW} | cut -d',' -f3)  # merged_accuracy_partial
    VALIDATION_ACC=$(echo ${LAST_ROW} | cut -d',' -f5)  # validation_accuracy

    # Count number of models merged (excluding header and base model)
    NUM_MERGED=$(($(wc -l < ${MERGE_LOG}) - 2))

    # Get final step number from results directories
    FINAL_STEP=$(find ${DIR}/results/merged_model -maxdepth 1 -type d -name "step_*" | sed 's/.*step_//' | sort -n | tail -1)

    # Handle empty values
    SELECTION_ACC=${SELECTION_ACC:-"N/A"}
    VALIDATION_ACC=${VALIDATION_ACC:-"N/A"}
    NUM_MERGED=${NUM_MERGED:-0}
    FINAL_STEP=${FINAL_STEP:-0}

    # Print formatted row
    printf "%-8s %-20s %-20s %-18s %-12s\n" "${BETA}" "${SELECTION_ACC}" "${VALIDATION_ACC}" "${NUM_MERGED}" "${FINAL_STEP}"
done

echo ""
echo "========================================="
echo ""
echo "To visualize results, run:"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME}              # Validation datasets (default)"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME} --selection  # Selection datasets"
echo ""
echo "Or view individual merge logs:"
for DIR in ${BETA_DIRS}; do
    BETA=$(basename ${DIR} | sed -n 's/.*beta_\([0-9.]*\).*/\1/p')
    echo "  beta=${BETA}: ${DIR}/merge_log.csv"
done
