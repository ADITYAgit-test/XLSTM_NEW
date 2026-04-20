#!/bin/bash

# Activate conda environment
source /home/bs_thesis/miniconda3/bin/activate final

# Experiment 9: 3D Orthogonal Slice Selection
# Dataset: Abdomen MR (702)
# Strategy: GM-ABS style (single middle axial slice)

set -e

echo "=========================================="
echo "Experiment 9: 3D Orthogonal Slice Selection"
echo "=========================================="

# Configuration
DATASET="702"
DATASET_NAME="Dataset702_AbdomenMR"
DIMENSION="3d_fullres"
TRAINER="nnUNetTrainerSliceSel"
LR="0.005"
BS="2"

# Paths
BASE_DIR="/home/bs_thesis/xlstm_new/xLSTM-UNet-PyTorch"
OUTPUT_DIR="$BASE_DIR/nnUNet_predictions/exp9_3D_orthogonal"
GT_PATH="$BASE_DIR/data/nnUNet_raw/${DATASET_NAME}/labelsTs"
EVAL_SCRIPT="$BASE_DIR/evaluation/abdomen_DSC_Eval.py"
NSD_SCRIPT="$BASE_DIR/evaluation/abdomen_NSD_Eval.py"
SLICE_FILE="$BASE_DIR/data/slice_selections/orthogonal_slices.json"

mkdir -p "$OUTPUT_DIR"

export SLICE_SELECTION_FILE="$SLICE_FILE"

# Set nnUNet environment variables
export nnUNet_raw="$BASE_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/data/nnUNet_results"

mkdir -p "$nnUNet_results"

# Training
echo "Starting training with orthogonal slice selection..."
echo "Using slice file: $SLICE_FILE"
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
    --disable_tta \
    -npp 1 -nps 1

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
echo "Experiment 9 Finished"
echo "=========================================="