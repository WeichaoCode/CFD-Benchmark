import os
import numpy as np
import logging
from datetime import datetime
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity


def call_compare_output_mismatch(llm_model, prompt_json):
    # === Config ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ground_truth_dir = os.path.join(ROOT_DIR, 'results/ground_truth')
    prediction_dir = os.path.join(ROOT_DIR, f'results/prediction/{llm_model}/{prompt_json}')

    # === Logging ===
    timestamp = datetime.now().strftime("%H-%M-%S")
    log_file = os.path.join(ROOT_DIR, f'compare/comparison_results_{llm_model}_{prompt_json}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear old logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup new log file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("====== Starting Comparison ======")

    # === Get common .npy files ===
    files_gt = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
    files_pred = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}
    common_files = sorted(files_gt & files_pred)

    print(f"Found {len(common_files)} common files to compare.")
    logging.info(f"Found {len(common_files)} common files.")

    def interpolate_to_match(gt, pred):
        if gt.shape == pred.shape:
            return pred
        try:
            factors = np.array(gt.shape) / np.array(pred.shape)
            pred_resized = zoom(pred, factors, order=1)
            return pred_resized
        except Exception as e:
            raise RuntimeError(f"Interpolation failed: {e}")

    def compute_losses(gt, pred):
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        mse = mean_squared_error(gt_flat, pred_flat)
        mae = mean_absolute_error(gt_flat, pred_flat)
        rmse = np.sqrt(mse)
        cosine_sim = cosine_similarity(gt_flat.reshape(1, -1), pred_flat.reshape(1, -1))[0][0]
        r2 = r2_score(gt_flat, pred_flat)
        return mse, mae, rmse, cosine_sim, r2

    results = {}

    for fname in common_files:
        try:
            gt_path = os.path.join(ground_truth_dir, fname)
            pred_path = os.path.join(prediction_dir, fname)
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            if gt.ndim == 1:
                gt = gt[:, np.newaxis]
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]

            pred = interpolate_to_match(gt, pred)

            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch after interpolation: {gt.shape} vs {pred.shape}")

            mse, mae, rmse, cosine_sim, r2 = compute_losses(gt, pred)

            results[fname] = {
                "MSE": f"{mse:.3e}",
                "MAE": f"{mae:.3e}",
                "RMSE": f"{rmse:.3e}",
                "CosineSimilarity": f"{cosine_sim:.3f}",
                "R2": f"{r2:.3f}"
            }

            logging.info(
                f"{fname}: MSE={mse:.3e}, MAE={mae:.3e}, RMSE={rmse:.3e}, Cosine={cosine_sim:.3f}, R2={r2:.3f}")

        except Exception as e:
            results[fname] = {"Error": str(e)}
            logging.error(f"❌ {fname} failed: {str(e)}")

    # === Print Summary ===
    print("\n=== Summary ===")
    for fname, res in results.items():
        print(f"📄 {fname}")
        for k, v in res.items():
            print(f"   {k}: {v}")
        print("-" * 40)

    print(f"\n🎯 Log saved to: {log_file}")


call_compare_output_mismatch("gpt-4o", "prompts")
call_compare_output_mismatch("o1-mini", "prompts")
call_compare_output_mismatch("sonnet-35", "prompts")
call_compare_output_mismatch("haiku", "prompts")
