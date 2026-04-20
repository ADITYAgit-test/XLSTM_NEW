#!/bin/bash

# Master script to run all 10 experiments sequentially

set -e

echo "=========================================="
echo "Starting All 10 Experiments"
echo "=========================================="

# Base paths
BASE_DIR="/home/bs_thesis/xlstm_new/xLSTM-UNet-PyTorch"
EVAL_DIR="$BASE_DIR/evaluation"

# Create results directory
# mkdir -p "$BASE_DIR/results"
# mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/nnUNet_predictions"

# Run each experiment sequentially
# echo "Running Experiment 1: 2D Full Supervision"
# bash "$BASE_DIR/experiments/run_exp1_2D_full.sh"

# echo "Running Experiment 2: 2D Random Selection"
# bash "$BASE_DIR/experiments/run_exp2_2D_random.sh"

# echo "Running Experiment 3: 2D Uniform Selection"
# bash "$BASE_DIR/experiments/run_exp3_2D_uniform.sh"

# echo "Running Experiment 4: 2D Orthogonal Selection"
# bash "$BASE_DIR/experiments/run_exp4_2D_orthogonal.sh"

# echo "Running Experiment 5: 2D Chunk + Hard Negative"
# bash "$BASE_DIR/experiments/run_exp5_2D_chunk_hard_negative.sh"

echo "Running Experiment 6: 3D Full Supervision"
bash "$BASE_DIR/experiments/run_exp6_3D_full.sh"

# echo "Running Experiment 7: 3D Random Selection"
# bash "$BASE_DIR/experiments/run_exp7_3D_random.sh"

# echo "Running Experiment 8: 3D Uniform Selection"
# bash "$BASE_DIR/experiments/run_exp8_3D_uniform.sh"

# echo "Running Experiment 9: 3D Orthogonal Selection"
# bash "$BASE_DIR/experiments/run_exp9_3D_orthogonal.sh"

# echo "Running Experiment 10: 3D Chunk + Hard Negative"
# bash "$BASE_DIR/experiments/run_exp10_3D_chunk_hard_negative.sh"

echo ""
echo "=========================================="
echo "All 10 Experiments Complete!"
echo "=========================================="