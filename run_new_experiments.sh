#!/usr/bin/env bash
# run_new_experiments.sh
# Runs all regression and multi-channel baseline experiments sequentially.
# Usage: bash Code/run_new_experiments.sh
# from the 04_digitalTwin/ directory, or adjust CODE_DIR below.

set -e
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$CODE_DIR/run_new.py"

echo "=== Regression baseline experiments (6 configs) ==="

REGRESSION_CONFIGS=(
    "experiments_regression/nasa_1dconv_regression/configs/config_regression_baseline.yaml"
    "experiments_regression/nasa_lstm_regression/configs/config_regression_baseline.yaml"
    "experiments_regression/nature_1dconv_regression/configs/config_regression_baseline.yaml"
    "experiments_regression/nature_lstm_regression/configs/config_regression_baseline.yaml"
    "experiments_regression/phm_1dconv_regression/configs/config_regression_baseline.yaml"
    "experiments_regression/phm_lstm_regression/configs/config_regression_baseline.yaml"
)

for cfg in "${REGRESSION_CONFIGS[@]}"; do
    full_path="$CODE_DIR/$cfg"
    echo ""
    echo "--- Running: $cfg ---"
    python "$RUN_SCRIPT" "$full_path"
done

echo ""
echo "=== Multi-channel baseline experiments (4 configs) ==="

MULTICHANNEL_CONFIGS=(
    "experiments_multichannel/nature_2ch_1dconv/configs/config_multichannel_baseline.yaml"
    "experiments_multichannel/nature_2ch_lstm/configs/config_multichannel_baseline.yaml"
    "experiments_multichannel/phm_3ch_1dconv/configs/config_multichannel_baseline.yaml"
    "experiments_multichannel/phm_3ch_lstm/configs/config_multichannel_baseline.yaml"
)

for cfg in "${MULTICHANNEL_CONFIGS[@]}"; do
    full_path="$CODE_DIR/$cfg"
    echo ""
    echo "--- Running: $cfg ---"
    python "$RUN_SCRIPT" "$full_path"
done

echo ""
echo "=== All 10 baseline experiments complete ==="
echo "Model outputs are in the respective experiment directories."
echo "Next: create zero-shot + TL configs using the trained model paths,"
echo "then run Wave 3E to fill Section 5.5 and 5.6 in Paper/main_restored.tex."
