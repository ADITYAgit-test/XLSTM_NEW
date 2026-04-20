"""
Slice Selection Experiment Configuration
==========================================

This module defines the experiment configurations for comparing 
different slice selection strategies under limited supervision.

Strategies:
1. full_volume - All slices (baseline)
2. random - Random B slices
3. uniform - Evenly spaced B slices  
4. orthogonal - GM-ABS style (1 axial slice)
5. chunk_hard_negative - Our method (tumor chunk + hard negatives)

Datasets:
- Dataset702: Abdomen MRI
"""

# Experiment Configuration
EXPERIMENTS = {
    "exp1_2D_full": {
        "name": "2D Full Supervision",
        "dimension": "2d",
        "strategy": "full_volume",
        "trainer": "nnUNetTrainerUxLSTMBot",
        "dataset": "702",
        "description": "Baseline: train on all slices (2D)"
    },
    "exp2_2D_random": {
        "name": "2D Random Selection",
        "dimension": "2d", 
        "strategy": "random",
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702",
        "description": "Random 10 slices per volume (2D)"
    },
    "exp3_2D_uniform": {
        "name": "2D Uniform Selection",
        "dimension": "2d",
        "strategy": "uniform", 
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702",
        "description": "Evenly spaced 10 slices (2D)"
    },
    "exp4_2D_orthogonal": {
        "name": "2D Orthogonal Selection",
        "dimension": "2d",
        "strategy": "orthogonal",
        "trainer": "nnUNetTrainerSliceSel", 
        "dataset": "702",
        "description": "GM-ABS style: single middle axial slice (2D)"
    },
    "exp5_2D_chunk_hard_negative": {
        "name": "2D Chunk + Hard Negative",
        "dimension": "2d",
        "strategy": "chunk_hard_negative",
        "k_tumor": 7,
        "m_negatives": 3,
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702", 
        "description": "Tumor chunk + hard negatives (2D)"
    },
    "exp6_3D_full": {
        "name": "3D Full Supervision",
        "dimension": "3d_fullres",
        "strategy": "full_volume",
        "trainer": "nnUNetTrainerUxLSTMBot",
        "dataset": "702",
        "description": "Baseline: train on all slices (3D)"
    },
    "exp7_3D_random": {
        "name": "3D Random Selection",
        "dimension": "3d_fullres",
        "strategy": "random",
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702",
        "description": "Random 10 slices per volume (3D)"
    },
    "exp8_3D_uniform": {
        "name": "3D Uniform Selection", 
        "dimension": "3d_fullres",
        "strategy": "uniform",
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702",
        "description": "Evenly spaced 10 slices (3D)"
    },
    "exp9_3D_orthogonal": {
        "name": "3D Orthogonal Selection",
        "dimension": "3d_fullres",
        "strategy": "orthogonal",
        "trainer": "nnUNetTrainerSliceSel",
        "dataset": "702",
        "description": "GM-ABS style: single middle axial slice (3D)"
    },
    "exp10_3D_chunk_hard_negative": {
        "name": "3D Chunk + Hard Negative",
        "dimension": "3d_fullres",
        "strategy": "chunk_hard_negative",
        "k_tumor": 7,
        "m_negatives": 3,
        "slice_budget": 10,
        "trainer": "nnUNetTrainerSliceSel", 
        "dataset": "702",
        "description": "Tumor chunk + hard negatives (3D)"
    }
}

# Hyperparameters (unchanged from baseline)
LEARNING_RATE = 0.005
BATCH_SIZE = 30

# Paths
BASE_DIR = "/home/bs_thesis/xlstm_new/xLSTM-UNet-PyTorch"
SLICE_SELECTIONS_DIR = f"{BASE_DIR}/data/slice_selections"
PREDICTIONS_DIR = f"{BASE_DIR}/nnUNet_predictions"
RESULTS_DIR = f"{BASE_DIR}/results"