"""
Name: compare_common_npy_files.py
Author: Your Name
Time: 2025-03-22
Description:
    Compare `.npy` files between two folders.
    Only compute metrics for files that appear in both folders (by filename).
    Log results including MSE, MAE, RMSE, Cosine Similarity, and R-squared.
"""

import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# === Config ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ground_truth_dir = os.path.join(ROOT_DIR, 'results/ground_truth')  # Folder A
prediction_dir = os.path.join(ROOT_DIR, 'results/prediction/o1-mini')  # Folder B

# === Logging Setup ===
timestamp = datetime.now().strftime("%H-%M-%S")
log_file = os.path.join(ROOT_DIR, f'compare/comparison_results_{timestamp}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("====== Starting Comparison of Common Files ======")

# === Gather file names ===
files_a = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
files_b = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}
common_files = sorted(files_a & files_b)

logging.info(f"Found {len(common_files)} common files to compare.")
print(f"Found {len(common_files)} common files to compare.")


# === Loss computation ===
def compute_losses(gt, pred):
    mse = mean_squared_error(gt, pred)
    mae = mean_absolute_error(gt, pred)
    rmse = np.sqrt(mse)
    cosine_sim = cosine_similarity(gt.reshape(1, -1), pred.reshape(1, -1))[0][0]
    r2 = r2_score(gt, pred)
    return mse, mae, rmse, cosine_sim, r2


# === Loop and compare ===
errors = {}

for file in common_files:
    gt_path = os.path.join(ground_truth_dir, file)
    pred_path = os.path.join(prediction_dir, file)

    try:
        gt_data = np.load(gt_path)
        pred_data = np.load(pred_path)

        # Handle shape mismatch by transposing prediction
        if gt_data.shape != pred_data.shape:
            pred_data = pred_data.T
            logging.warning(f"Shape mismatch in {file} — tried transpose. New shape: {pred_data.shape}")

        # If still mismatched, skip
        if gt_data.shape != pred_data.shape:
            raise ValueError(f"Still shape mismatch after transpose: GT {gt_data.shape}, PRED {pred_data.shape}")

        # Compute metrics
        mse, mae, rmse, cosine_sim, r2 = compute_losses(gt_data, pred_data)

        # Save results
        errors[file] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Cosine Similarity': cosine_sim,
            'R-squared': r2
        }

        logging.info(f"✅ {file} comparison completed.")
        logging.info(
            f"Results for {file}: MSE={mse}, MAE={mae}, RMSE={rmse}, "
            f"Cosine Similarity={cosine_sim}, R-squared={r2}"
        )

    except Exception as e:
        errors[file] = {'Error': str(e)}
        logging.error(f"❌ Error comparing {file}: {str(e)}")

# === Print results summary ===
logging.info("====== Comparison Completed ======")
print("\n=== Results ===")
for file, result in errors.items():
    if 'Error' in result:
        print(f"❌ {file}: {result['Error']}")
    else:
        print(f"✅ {file}:")
        for k, v in result.items():
            print(f"   {k}: {v:.6f}")
    print("-" * 40)

print(f"\n🎯 Comparison completed. Log saved to: {log_file}")
