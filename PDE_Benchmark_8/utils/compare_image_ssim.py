import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# === Configuration ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the directories containing the images
save_dir_gt = os.path.join(ROOT_DIR, 'compare_images/ground_truth')
save_dir_pred = os.path.join(ROOT_DIR, 'compare_images/prediction')
save_csv_path = os.path.join(ROOT_DIR, 'compare_images/table')
os.makedirs(save_csv_path, exist_ok=True)


# === Helper Functions ===
def get_problem_name_pred(filename):
    return filename.rsplit('_', 1)[0]


def get_problem_name_gt(filename):
    return filename.rsplit('_', 2)[0]


def calculate_metrics(img1, img2):
    # Resize img2 to match img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # SSIM
    ssim_index, _ = ssim(gray1, gray2, full=True)

    # MSE
    mse = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)

    # MAE
    mae = np.mean(np.abs(gray1.astype("float") - gray2.astype("float")))

    # PSNR
    psnr = cv2.PSNR(gray1, gray2)

    return ssim_index, psnr, mse, mae


# === Find common files ===
gt_files = {get_problem_name_gt(f): f for f in os.listdir(save_dir_gt) if f.endswith('.png')}
pred_files = {get_problem_name_pred(f): f for f in os.listdir(save_dir_pred) if f.endswith('.png')}

common_keys = gt_files.keys() & pred_files.keys()

# === Compute Metrics for Common Files ===
results = []
for problem_name in sorted(common_keys):
    gt_path = os.path.join(save_dir_gt, gt_files[problem_name])
    pred_path = os.path.join(save_dir_pred, pred_files[problem_name])

    try:
        img_gt = cv2.imread(gt_path)
        img_pred = cv2.imread(pred_path)

        ssim_score, psnr_score, mse_score, mae_score = calculate_metrics(img_gt, img_pred)

        results.append({
            "Problem Name": problem_name,
            "SSIM": round(ssim_score, 4),
            "PSNR": round(psnr_score, 2),
            "MSE": round(mse_score, 4),
            "MAE": round(mae_score, 4)
        })
    except Exception as e:
        print(f"❌ Error comparing {problem_name}: {str(e)}")

# === Save to CSV ===
df = pd.DataFrame(results)
csv_file = os.path.join(save_csv_path, 'image_similarity_scores.csv')
df.to_csv(csv_file, index=False)

print(f"✅ Image comparison complete. Table saved to:\n{csv_file}")
