import os
import numpy as np
import matplotlib.pyplot as plt


def call_show_image(llm_model, prompts_json):
    # === Configuration ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ground_truth_dir = os.path.join(ROOT_DIR, 'results/solution')
    prediction_dir = os.path.join(ROOT_DIR, f'results/prediction/{llm_model}/{prompts_json}')
    save_dir = os.path.join(ROOT_DIR, f'image/{llm_model}/{prompts_json}')
    os.makedirs(save_dir, exist_ok=True)

    # === List .npy files in both directories ===
    gt_files = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
    pred_files = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}

    # === Common files (files that exist in both directories) ===
    common_files = gt_files.intersection(pred_files)

    # === Plotting Functions ===

    def plot_1d(gt, pred, file_name):
        x_gt = np.arange(len(gt))
        x_pred = np.arange(len(pred))

        plt.figure(figsize=(10, 6))
        plt.suptitle(f"{file_name}")
        plt.subplot(2, 1, 1)
        plt.plot(x_gt, gt, label='Ground Truth', color='blue')
        plt.title('Ground Truth')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(x_pred, pred, label='Prediction', color='green')
        plt.title('Prediction')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
        plt.close()

    def plot_2d(gt, pred, file_name):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plt.suptitle(f"{file_name}")
        im0 = axes[0].imshow(gt, cmap='viridis', origin='lower')
        axes[0].set_title('Ground Truth')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred, cmap='viridis', origin='lower')
        axes[1].set_title('Prediction')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
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

    print(f"\n🎯 Plotting complete. Images saved to: {save_dir}")

#
# call_show_image("sonnet-35", "prompts")
# call_show_image("haiku", "prompts")

call_show_image("o1-mini", "prompts")
