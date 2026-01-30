#!/bin/bash
# Analysis script for EMA vs Weight Averaging sweep results
# Usage: ./analyze_method_sweep.sh [experiment_name]

EXPERIMENT_NAME=${1:-"ema_holdout"}
BASE_DIR="outputs/method_sweep"

echo "Analyzing method sweep results for: ${EXPERIMENT_NAME}"
echo "========================================="
echo ""

if [ ! -d "${BASE_DIR}" ]; then
    echo "Error: Results directory ${BASE_DIR} not found"
    exit 1
fi

METHOD_DIRS=$(find ${BASE_DIR} -maxdepth 1 -type d \( -name "${EXPERIMENT_NAME}_ema" -o -name "${EXPERIMENT_NAME}_weight_averaging" \) | sort)

if [ -z "${METHOD_DIRS}" ]; then
    echo "Error: No results found for experiment ${EXPERIMENT_NAME}"
    echo "Expected directories: ${BASE_DIR}/${EXPERIMENT_NAME}_ema, ${BASE_DIR}/${EXPERIMENT_NAME}_weight_averaging"
    exit 1
fi

echo "Found results for the following methods:"
for DIR in ${METHOD_DIRS}; do
    METHOD=$(basename ${DIR} | sed "s/${EXPERIMENT_NAME}_//")
    echo "  - ${METHOD}: ${DIR}"
done
echo ""

echo ""
echo "Results Summary:"
echo "----------------"
printf "%-20s %-20s %-20s %-18s %-12s\n" "Method" "Selection Acc" "Validation Acc" "Models Merged" "Final Step"
printf "%-20s %-20s %-20s %-18s %-12s\n" "------" "--------------" "---------------" "-------------" "----------"

for DIR in ${METHOD_DIRS}; do
    METHOD=$(basename ${DIR} | sed "s/${EXPERIMENT_NAME}_//")
    MERGE_LOG="${DIR}/merge_log.csv"

    if [ ! -f "${MERGE_LOG}" ]; then
        echo "Warning: No merge_log.csv found in ${DIR}"
        continue
    fi

    LAST_ROW=$(tail -n 1 ${MERGE_LOG})
    SELECTION_ACC=$(echo ${LAST_ROW} | cut -d',' -f3)
    VALIDATION_ACC=$(echo ${LAST_ROW} | cut -d',' -f5)
    NUM_MERGED=$(($(wc -l < ${MERGE_LOG}) - 2))
    FINAL_STEP=$(find ${DIR}/results/merged_model -maxdepth 1 -type d -name "step_*" 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)

    SELECTION_ACC=${SELECTION_ACC:-"N/A"}
    VALIDATION_ACC=${VALIDATION_ACC:-"N/A"}
    NUM_MERGED=${NUM_MERGED:-0}
    FINAL_STEP=${FINAL_STEP:-0}

    printf "%-20s %-20s %-20s %-18s %-12s\n" "${METHOD}" "${SELECTION_ACC}" "${VALIDATION_ACC}" "${NUM_MERGED}" "${FINAL_STEP}"
done

echo ""
echo "========================================="
echo ""
echo "To visualize results, run:"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME} --base-dir outputs/method_sweep              # Validation datasets (default)"
echo "  python plotting/plot_sweep_results.py ${EXPERIMENT_NAME} --base-dir outputs/method_sweep --selection  # Selection datasets"
echo ""
echo "Or view individual merge logs:"
for DIR in ${METHOD_DIRS}; do
    METHOD=$(basename ${DIR} | sed "s/${EXPERIMENT_NAME}_//")
    echo "  ${METHOD}: ${DIR}/merge_log.csv"
done
