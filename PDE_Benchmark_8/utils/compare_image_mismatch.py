import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd


# === Helper Functions ===
def get_problem_name_pred(filename):
    return filename.rsplit('_', 1)[0]


def get_problem_name_gt(filename):
    return filename.rsplit('_', 2)[0]


def call_compare_image_mismatch(llm_model, prompts_json):
    # === Configuration ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define the directories containing the images
    save_dir_gt = os.path.join(ROOT_DIR, f'compare_images/ground_truth/{llm_model}/{prompts_json}')
    save_dir_pred = os.path.join(ROOT_DIR, f'compare_images/prediction/{llm_model}/{prompts_json}')
    save_csv_path = os.path.join(ROOT_DIR, f'compare_images/table/{llm_model}/{prompts_json}')
    os.makedirs(save_csv_path, exist_ok=True)

    # === Find common files ===
    gt_files = {get_problem_name_gt(f): f for f in os.listdir(save_dir_gt) if f.endswith('.png')}
    pred_files = {get_problem_name_pred(f): f for f in os.listdir(save_dir_pred) if f.endswith('.png')}

    common_files = gt_files.keys() & pred_files.keys()

    results = []

    for filename in common_files:
        gt_path = os.path.join(save_dir_gt, gt_files[filename])
        pred_path = os.path.join(save_dir_pred, pred_files[filename])

        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        img_pred_resized = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        abs_error = np.abs(img_gt - img_pred_resized)
        mse_val = np.mean((img_gt - img_pred_resized) ** 2)
        mae_val = np.mean(abs_error)
        ssim_val = ssim(img_gt, img_pred_resized, data_range=1.0)
        psnr_val = psnr(img_gt, img_pred_resized, data_range=1.0)

        results.append({
            "filename": filename,
            "MSE": mse_val,
            "MAE": mae_val,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    csv_file = os.path.join(save_csv_path, 'image_similarity_scores.csv')
    df.to_csv(csv_file, index=False)


call_compare_image_mismatch("gpt-4o", "prompts")
call_compare_image_mismatch("o1-mini", "prompts")
call_compare_image_mismatch("sonnet-35", "prompts")
call_compare_image_mismatch("haiku", "prompts")
