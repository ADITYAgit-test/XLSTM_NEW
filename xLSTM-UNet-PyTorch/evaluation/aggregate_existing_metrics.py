import os
import pandas as pd

def aggregate_metrics(pred_root, output_file):
    results = []
    
    exp_folders = [f for f in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, f))]
    exp_folders.sort()
    
    for exp in exp_folders:
        metrics_dir = os.path.join(pred_root, exp, 'metrics')
        dsc_file = os.path.join(metrics_dir, 'DSC.csv')
        nsd_file = os.path.join(metrics_dir, 'NSD.csv')
        
        if os.path.exists(dsc_file) and os.path.exists(nsd_file):
            try:
                df_dsc = pd.read_csv(dsc_file)
                df_nsd = pd.read_csv(nsd_file)
                
                # Exclude 'Name' column for averaging
                dsc_cols = [c for c in df_dsc.columns if c != 'Name']
                nsd_cols = [c for c in df_nsd.columns if c != 'Name']
                
                # Calculate mean per organ across all cases, then mean across all organs
                avg_dsc = df_dsc[dsc_cols].mean().mean()
                avg_nsd = df_nsd[nsd_cols].mean().mean()
                
                results.append({
                    'Experiment': exp,
                    'Avg_DSC': round(avg_dsc, 4),
                    'Avg_NSD': round(avg_nsd, 4)
                })
                print(f"Aggregated {exp}: DSC={avg_dsc:.4f}, NSD={avg_nsd:.4f}")
            except Exception as e:
                print(f"Error processing {exp}: {e}")
        else:
            print(f"Metrics not found for {exp} in {metrics_dir}")
            
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary saved to {output_file}")
        print(summary_df.to_string(index=False))
    else:
        print("No metrics aggregated.")

if __name__ == '__main__':
    pred_root = 'xLSTM-UNet-PyTorch/nnUNet_predictions'
    output_file = 'summary_metrics.csv'
    aggregate_metrics(pred_root, output_file)
