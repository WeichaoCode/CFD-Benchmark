import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# === Configuration ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the directories containing the images
save_dir_gt = os.path.join(ROOT_DIR, 'compare_images/ground_truth')
save_dir_pred = os.path.join(ROOT_DIR, 'compare_images/prediction')
save_csv_path = os.path.join(ROOT_DIR, 'compare_images/table')

# Initialize an empty list to store the SSIM results
ssim_results = []


# Function to extract the problem name by ignoring the last part after the last '_'
def get_problem_name_pred(filename):
    return filename.rsplit('_', 1)[0]  # Split from the right at the last '_'


def get_problem_name_gt(filename):
    return filename.rsplit('_', 2)[0]  # Split from the right at the last '_'


# Get the list of files in both folders
files1 = {get_problem_name_gt(f) for f in os.listdir(save_dir_gt) if f.endswith('.png')}
files2 = {get_problem_name_pred(f) for f in os.listdir(save_dir_pred) if f.endswith('.png')}

# Find common files between the two folders
common_files = files1.intersection(files2)

# Loop over each common file and calculate SSIM
for file_name in common_files:
    # Get the full filenames by appending the suffixes
    gt_file_name = file_name + '_ground_truth.png'
    pred_file_name = file_name + '_prediction.png'

    # Load the images using the full file names
    image1 = cv2.imread(os.path.join(save_dir_gt, gt_file_name))
    image2 = cv2.imread(os.path.join(save_dir_pred, pred_file_name))

    # Resize image2 to the size of image1
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convert images to grayscale for SSIM
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_index, _ = ssim(image1_gray, image2_gray, full=True)

    # Store the result in the list
    ssim_results.append([file_name, ssim_index])

# Convert the results into a pandas DataFrame
ssim_df = pd.DataFrame(ssim_results, columns=['File Name', 'SSIM'])

# Save the DataFrame to a CSV file
ssim_df.to_csv(f'{save_csv_path}/ssim_results.csv', index=False)

print("SSIM results saved to ssim_results.csv")
