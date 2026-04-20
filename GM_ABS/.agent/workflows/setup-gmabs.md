---
description: Pipeline for setting up GM-ABS environment, data, and models
---

# GM-ABS Setup Pipeline

Follow these steps to set up the environment, prepare the data, and install required models.

## 1. Environment Setup
Install the base requirements and the MobileSAM package.

// turbo
```bash
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## 2. Data Preparation
Reorganize the raw 2018LA dataset and generate cross-labels.

// turbo
```bash
cd GM-ABS/code_GMABS/dataloaders
# Reorganize raw data into H5 format
python resort_LAdata.py
# Generate cross-labeled dataset (3 slices per scan)
python cross_labeling.py
cd ../..
```

## 3. Model Weights
Ensure pre-trained weights are in the correct directory.

```bash
mkdir -p GM-ABS/code_GMABS/sam_weights
# Ensure mobile_sam.pt, medsam_vit_b.pth, and lite_medsam.pth are in this folder.
```

## 4. Run Training
Execute the training script with the desired parameters.

```bash
cd GM-ABS/code_GMABS
python train_final_GMABS_LA_public.py \
    --labeled_num 4 \
    --budget 16 \
    --active_type uncerper_div \
    --gpu 0 \
    --label_strategy majority \
    --exp LA_GMABS_HERD \
    --add_point 2
```
