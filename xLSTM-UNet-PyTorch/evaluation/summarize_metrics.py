import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

def find_lower_upper_zbound(organ_mask):
    organ_mask = np.uint8(organ_mask)
    if np.max(organ_mask) == 0:
        return 0, 0
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    return z_lower, z_upper

def evaluate_folder(gt_path, seg_path, save_path):
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()

    if not filenames:
        print(f"No .nii.gz files found in {seg_path}")
        return None

    label_tolerance = OrderedDict({
        'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5,
        'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
        'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3
    })

    seg_metrics = OrderedDict()
    seg_metrics['Name'] = list()
    for organ in label_tolerance.keys():
        seg_metrics[f'{organ}_DSC'] = list()
        seg_metrics[f'{organ}_NSD'] = list()

    os.makedirs(save_path, exist_ok=True)
    
    for name in tqdm(filenames, desc=f"Evaluating {basename(seg_path)}"):
        seg_metrics['Name'].append(name)
        gt_file = join(gt_path, name)
        seg_file = join(seg_path, name)
        
        if not os.path.exists(gt_file):
            print(f"Warning: GT file {gt_file} not found. Skipping.")
            for organ in label_tolerance.keys():
                seg_metrics[f'{organ}_DSC'].append(0)
                seg_metrics[f'{organ}_NSD'].append(0)
            continue

        gt_nii = nb.load(gt_file)
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(seg_file).get_fdata())

        for i, (organ, tolerance) in enumerate(label_tolerance.items(), 1):
            if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
                DSC_i = 1.0
                NSD_i = 1.0
            elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
                DSC_i = 0.0
                NSD_i = 0.0
            else:
                if i in [5, 6, 10]:  # Aorta, IVC, Esophagus
                    z_lower, z_upper = find_lower_upper_zbound(gt_data == i)
                    organ_i_gt = gt_data[:, :, z_lower:z_upper+1] == i
                    organ_i_seg = seg_data[:, :, z_lower:z_upper+1] == i
                else:
                    organ_i_gt = gt_data == i
                    organ_i_seg = seg_data == i

                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                if np.isnan(DSC_i): DSC_i = 0.0
                
                surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, tolerance)
            
            seg_metrics[f'{organ}_DSC'].append(round(float(DSC_i), 4))
            seg_metrics[f'{organ}_NSD'].append(round(float(NSD_i), 4))

    df = pd.DataFrame(seg_metrics)
    df.to_csv(join(save_path, 'metrics_per_case.csv'), index=False)
    
    # Calculate averages
    dsc_cols = [c for c in df.columns if c.endswith('_DSC')]
    nsd_cols = [c for c in df.columns if c.endswith('_NSD')]
    
    avg_dsc = df[dsc_cols].mean().mean()
    avg_nsd = df[nsd_cols].mean().mean()
    
    return avg_dsc, avg_nsd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth labels')
    parser.add_argument('--pred_root', type=str, required=True, help='Path to directory containing experiment folders')
    parser.add_argument('--output_file', type=str, default='summary_metrics.csv', help='Path to save summary results')
    args = parser.parse_args()

    results = []
    exp_folders = [f for f in os.listdir(args.pred_root) if os.path.isdir(join(args.pred_root, f))]
    exp_folders.sort()

    for exp in exp_folders:
        seg_path = join(args.pred_root, exp)
        save_path = seg_path # Save per-case metrics inside the experiment folder
        
        print(f"\nProcessing experiment: {exp}")
        avg_metrics = evaluate_folder(args.gt_path, seg_path, save_path)
        
        if avg_metrics:
            avg_dsc, avg_nsd = avg_metrics
            results.append({
                'Experiment': exp,
                'Avg_DSC': round(avg_dsc, 4),
                'Avg_NSD': round(avg_nsd, 4)
            })
            print(f"Result for {exp}: DSC={avg_dsc:.4f}, NSD={avg_nsd:.4f}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(args.output_file, index=False)
        print(f"\nSummary saved to {args.output_file}")
        print(summary_df.to_string(index=False))
