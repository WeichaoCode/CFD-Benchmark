import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the JSON files
with open("/opt/CFD-Benchmark/data/output_true.json", "r") as file:
    true_data = json.load(file)

with open("/opt/CFD-Benchmark/data/output_generate.json", "r") as file:
    generated_data = json.load(file)

# Create a directory for saving plots
output_dir = "/opt/CFD-Benchmark/comparison_plots"
os.makedirs(output_dir, exist_ok=True)


# Function to compute Mean Squared Error (MSE)
def compute_mse(true_values, generated_values):
    return np.mean((np.array(true_values) - np.array(generated_values)) ** 2)


# Iterate over filenames that exist in both JSON files
for filename in true_data.keys():
    if filename in generated_data:
        true_case = true_data[filename]
        generated_case = generated_data[filename]

        # Iterate over each key in the file's data (e.g., u, v, p)
        for key in true_case.keys():
            if key in generated_case:
                true_values = true_case[key]
                generated_values = generated_case[key]

                # Check if both are lists (numerical arrays)
                if isinstance(true_values, list) and isinstance(generated_values, list):
                    if len(true_values) == len(generated_values):  # Ensure same length
                        # Compute loss (MSE)
                        loss = compute_mse(true_values, generated_values)

                        # Plot comparison
                        plt.figure(figsize=(8, 6))
                        plt.plot(true_values, label="True Output", linestyle="dashed", color="blue")
                        plt.plot(generated_values, label="Generated Output", linestyle="solid", color="red")
                        plt.xlabel("Index")
                        plt.ylabel(key)
                        plt.title(f"{filename} - {key}\nMSE Loss: {loss:.6f}")
                        plt.legend()
                        plt.grid()

                        # Save the plot
                        plot_filename = os.path.join(output_dir, f"{filename}_{key}.png")
                        plt.savefig(plot_filename)
                        plt.close()

                        print(f"Saved comparison plot: {plot_filename}")
                    else:
                        print(f"Warning: Data length mismatch for {filename} -> {key}")

print("All comparisons completed.")
