import os
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ground_truth_dir = os.path.join(ROOT_DIR, 'results/ground_truth')
prediction_dir = os.path.join(ROOT_DIR, 'results/prediction')
save_dir = os.path.join(ROOT_DIR, 'image')
os.makedirs(save_dir, exist_ok=True)

# === List .npy files ===
files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')]


# === Plotting Functions ===

def plot_1d(gt, pred, file_name):
    x = np.arange(len(gt))
    error = np.abs(gt - pred)
    mse = np.mean((gt - pred) ** 2)

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"{file_name} MSE: {mse}")
    plt.subplot(3, 1, 1)
    plt.plot(x, gt, label='Ground Truth', color='blue')
    plt.title('Ground Truth')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x, pred, label='Prediction', color='green')
    plt.title('Prediction')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x, error, label='Absolute Error', color='red')
    plt.title('Absolute Error')
    plt.xlabel('Index')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


def plot_2d(gt, pred, file_name):
    error = np.abs(gt - pred)
    mse = np.mean((gt - pred) ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"{file_name} MSE: {mse}")
    im0 = axes[0].imshow(gt, cmap='viridis', origin='lower')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred, cmap='viridis', origin='lower')
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, cmap='hot', origin='lower')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


# === Iterate and Plot ===
for file in files:
    gt_path = os.path.join(ground_truth_dir, file)
    pred_path = os.path.join(prediction_dir, file)

    try:
        gt = np.load(gt_path)
        pred = np.load(pred_path)

        # Auto-transpose if needed
        if gt.shape != pred.shape:
            pred = pred.T

        # Plot
        if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
            plot_1d(gt.flatten(), pred.flatten(), file.replace(".npy", ""))
        elif gt.ndim == 2:
            plot_2d(gt, pred, file.replace(".npy", ""))
        else:
            print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
    except Exception as e:
        print(f"❌ Error plotting {file}: {str(e)}")

print(f"\n🎯 Plotting complete. Images saved to: {save_dir}")
