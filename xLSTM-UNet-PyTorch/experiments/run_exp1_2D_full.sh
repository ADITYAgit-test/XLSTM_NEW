#!/bin/bash

# Activate conda environment
source /home/bs_thesis/miniconda3/bin/activate final

# Experiment 1: 2D Full Supervision Baseline
# Dataset: Abdomen MR (702)
# Strategy: Full supervision (all slices)

set -e

echo "=========================================="
echo "Experiment 1: 2D Full Supervision"
echo "=========================================="

# Configuration
DATASET="702"
DATASET_NAME="Dataset702_AbdomenMR"
DIMENSION="2d"
TRAINER="nnUNetTrainerUxLSTMBot"
LR="0.005"
BS="30"

# Paths
BASE_DIR="/home/bs_thesis/xlstm_new/xLSTM-UNet-PyTorch"
OUTPUT_DIR="$BASE_DIR/nnUNet_predictions/exp1_2D_full"
GT_PATH="$BASE_DIR/data/nnUNet_raw/${DATASET_NAME}/labelsTs"
EVAL_SCRIPT="$BASE_DIR/evaluation/abdomen_DSC_Eval.py"
NSD_SCRIPT="$BASE_DIR/evaluation/abdomen_NSD_Eval.py"

mkdir -p "$OUTPUT_DIR"

# Set nnUNet environment variables
export nnUNet_raw="$BASE_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/data/nnUNet_results"

mkdir -p "$nnUNet_results"

# Training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train ${DATASET} ${DIMENSION} all \
    -tr ${TRAINER} \
    -lr ${LR} \
    -bs ${BS}

echo "Training complete"

# Prediction
echo "Starting prediction..."
nnUNetv2_predict \
    -i $BASE_DIR/data/nnUNet_raw/${DATASET_NAME}/imagesTs \
    -o $OUTPUT_DIR \
    -d ${DATASET} \
    -c ${DIMENSION} \
    -f all \
    -tr ${TRAINER} \
    --disable_tta

echo "Prediction complete"

# Evaluation
echo "Starting evaluation..."
export GT_PATH=$GT_PATH
export SEG_PATH=$OUTPUT_DIR
export SAVE_PATH=$OUTPUT_DIR/metrics
mkdir -p $SAVE_PATH

python ${EVAL_SCRIPT} --gt_path $GT_PATH --seg_path $SEG_PATH --save_path $SAVE_PATH
python ${NSD_SCRIPT} --gt_path $GT_PATH --seg_path $SEG_PATH --save_path $SAVE_PATH

echo "Evaluation complete"
echo "=========================================="
echo "Experiment 1 Finished"
echo "=========================================="