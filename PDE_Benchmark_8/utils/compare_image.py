import os
import numpy as np
import matplotlib.pyplot as plt
# === Configuration ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ground_truth_dir = os.path.join(ROOT_DIR, 'results/solution')
prediction_dir = os.path.join(ROOT_DIR, 'results/prediction/gpt-4o/prompts')
save_dir_gt = os.path.join(ROOT_DIR, 'compare_images/ground_truth')
save_dir_pred = os.path.join(ROOT_DIR, 'compare_images/prediction')
os.makedirs(save_dir_gt, exist_ok=True)
os.makedirs(save_dir_pred, exist_ok=True)

# === List .npy files in both directories ===
gt_files = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
pred_files = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}

# === Common files (files that exist in both directories) ===
common_files = gt_files.intersection(pred_files)


# === Plotting Functions ===

def plot_1d(gt, pred, file_name):
    x_gt = np.arange(len(gt))
    x_pred = np.arange(len(pred))

    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_gt, gt, label='Ground Truth', color='blue')
    plt.title(f'Ground Truth - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_pred, pred, label='Prediction', color='green')
    plt.title(f'Prediction - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


def plot_2d(gt, pred, file_name):

    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    im0 = plt.imshow(gt, cmap='viridis', origin='lower')
    plt.title(f'Ground Truth - {file_name}')
    plt.colorbar(im0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    im1 = plt.imshow(pred, cmap='viridis', origin='lower')
    plt.title(f'Prediction - {file_name}')
    plt.colorbar(im1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


# === Iterate and Plot for Common Files ===
for file in common_files:
    gt_path = os.path.join(ground_truth_dir, file)
    pred_path = os.path.join(prediction_dir, file)

    try:
        gt = np.load(gt_path)
        pred = np.load(pred_path)

        # Plot
        if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
            plot_1d(gt.flatten(), pred.flatten(), file.replace(".npy", ""))
        elif gt.ndim == 2:
            plot_2d(gt, pred, file.replace(".npy", ""))
        else:
            print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
    except Exception as e:
        print(f"❌ Error plotting {file}: {str(e)}")

print(f"\n🎯 Plotting complete. Images saved to: {save_dir_gt} and {save_dir_pred}")