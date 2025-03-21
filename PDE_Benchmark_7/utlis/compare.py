import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root

# Set up logging configuration
timestamp = datetime.now().strftime("%H-%M-%S")  # Time-stamped log file name
log_file = os.path.join(ROOT_DIR, f'compare/comparison_results_{timestamp}.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("====== Starting Comparison Process ======")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root

# Define the directories for ground truth and predictions
ground_truth_dir = os.path.join(ROOT_DIR, 'results/ground_truth')  # Replace with your ground truth directory
prediction_dir = os.path.join(ROOT_DIR, 'results/prediction')  # Replace with your prediction directory

# List all .npy files in the ground truth directory without sorting (in the order they appear)
ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')]

# List all .npy files in the prediction directory (same as above)
prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.npy')]

# Ensure both directories have the same files in the same order
if len(ground_truth_files) != len(prediction_files):
    logging.error("The number of files in the ground truth and prediction directories do not match.")
    print("Warning: The number of files in the ground truth and prediction directories do not match.")
    exit()

# If needed, you can verify that the order is the same (optional)
for i in range(len(ground_truth_files)):
    if ground_truth_files[i] != prediction_files[i]:
        logging.error(f"Files in the ground truth and prediction directories are not in the same order at index {i}.")
        print("Warning: The files in the ground truth and prediction directories are not in the same order.")
        exit()

logging.info(f"Found {len(ground_truth_files)} matching files. Proceeding with comparison...")

# Define a dictionary to store the errors for each file
errors = {}


# Define loss functions
def compute_losses(gt, pred):
    # MSE
    mse = mean_squared_error(gt, pred)
    # MAE
    mae = mean_absolute_error(gt, pred)
    # RMSE
    rmse = np.sqrt(mse)
    # Cosine Similarity
    cosine_sim = cosine_similarity(gt.reshape(1, -1), pred.reshape(1, -1))[0][0]
    # R-squared
    r2 = r2_score(gt, pred)

    return mse, mae, rmse, cosine_sim, r2


# Iterate through each .npy file and compute losses
for file in ground_truth_files:
    gt_file_path = os.path.join(ground_truth_dir, file)
    pred_file_path = os.path.join(prediction_dir, file)

    # Try to load the ground truth and prediction data
    try:
        gt_data = np.load(gt_file_path)
        pred_data = np.load(pred_file_path)

        # Ensure both arrays have the same shape
        if gt_data.shape != pred_data.shape:
            raise ValueError(
                f"Shape mismatch between {file}. Ground truth shape: {gt_data.shape}, Prediction shape: {pred_data.shape}")

        # Compute the loss functions
        mse, mae, rmse, cosine_sim, r2 = compute_losses(gt_data, pred_data)

        # Store the errors
        errors[file] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Cosine Similarity': cosine_sim,
            'R-squared': r2
        }

        logging.info(f"Success: Comparison for {file} completed.")
        logging.info(
            f"Results for {file}: MSE={mse}, MAE={mae}, RMSE={rmse}, Cosine Similarity={cosine_sim}, R-squared={r2}")

    except Exception as e:
        errors[file] = {'Error': str(e)}
        logging.error(f"Error comparing {file}: {str(e)}")

# Final report logging
logging.info("====== Comparison Completed ======")

# Print the results for each file (for visual feedback)
for file, error in errors.items():
    if 'Error' in error:
        print(f"❌ {file}: Error - {error['Error']}")
    else:
        print(f"✅ {file}:")
        for loss_name, loss_value in error.items():
            print(f"  {loss_name}: {loss_value:.6f}")
    print("-" * 50)

print(f"\n🎯 Comparison completed. Results saved in: {log_file}")
