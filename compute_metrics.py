import os
import re
import glob
import globals
import argparse
import pandas as pd

from evaluation_metrics import compute_metrics_from_file

# -------------------------------
# Main loop: iterate through files, compute metrics, and save summary CSV
# -------------------------------
if __name__ == '__main__':

    # Parse command line arguments to get the path to the desired dataset results.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',   '--path',  help='Define the path to the desired dataset results.', default=None)
    
    parsed_args = parser.parse_args()
    results_path         = parsed_args.path

    if results_path is not None: 
        if not os.path.exists(results_path):
            print(f"Path {results_path} does not exist. Please provide a valid path.")
            exit()

    else:
        print("No path provided. Please provide a valid path. e.g. './COF_CH4_H2_Keskin_NCH4/random_sampling_plots'")
        exit()

    file_pattern = os.path.join(results_path, 'results_with_bo_predictions_*_samples.csv')

    # Use glob to match files with the pattern. Adjust the pattern if needed.
    files = glob.glob(file_pattern)
    
    all_metrics = []
    for file in files:
        # Extract sample size from filename (e.g., the number between "predictions_" and "_samples")
        match = re.search(r'results_with_bo_predictions_(\d+)_samples\.csv', file)
        if match:
            sample_size = int(match.group(1))
        else:
            sample_size = None
        
        # Compute metrics for the file
        metrics = compute_metrics_from_file(file)
        metrics['sample_size'] = sample_size
        all_metrics.append(metrics)
        print(f"Processed file: {file} with sample size {sample_size}")

    if not all_metrics:
        print("No files found containinig bo predictions. Please check the provided filepath.")
        exit()

    # Create a summary DataFrame and save to CSV
    summary_df   = pd.DataFrame(all_metrics)
    save_path = os.path.join(globals.save_dir,"evaluation_metrics_results" + '.csv')

    summary_df.to_csv(save_path, index=False)
    print("Saved evaluation metrics to evaluation_metrics_summary.csv")
