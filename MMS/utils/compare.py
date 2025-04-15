import json
import numpy as np
import matplotlib.pyplot as plt
import os


def compare_with_exact_solution(input_json, output_folder, comparison_json):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the data from JSON file
    with open(input_json, "r") as file:
        data = json.load(file)

    comparison_results = {}

    # Loop through each entry in the JSON
    for filename, content in data.items():
        u_values = np.array(content["u"])

        # Generate x values assuming uniform distribution in [0, 2]
        x = np.linspace(0, 2, len(u_values))

        # Exact solution at t = 2
        exact_solution = np.exp(-2) * np.sin(np.pi * x)

        # Calculate MSE and L2 loss
        mse = np.mean((u_values - exact_solution) ** 2)
        l2_loss = np.sqrt(np.sum((u_values - exact_solution) ** 2))

        # Save the comparison results in JSON format
        comparison_results[filename] = {
            "MSE": mse,
            "L2_loss": l2_loss
        }

        # Plotting
        plt.figure()
        plt.plot(x, u_values, label="Numerical Solution", marker='o')
        plt.plot(x, exact_solution, label="Exact Solution", linestyle='--')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Comparison for {filename}")
        plt.legend()
        plt.grid()

        # Save the plot
        plot_path = os.path.join(output_folder, f"{filename}_comparison.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Processed {filename}: MSE = {mse:.6f}, L2 Loss = {l2_loss:.6f}")

    # Save comparison results to compare.json
    with open(comparison_json, "w") as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\nComparison results saved in {comparison_json}")
    print(f"Plots saved in {output_folder}")


# Example usage
input_json = "/opt/CFD-Benchmark/MMS/result/output.json"  # Input JSON file
output_folder = "/opt/CFD-Benchmark/MMS/result"  # Folder to save the plots
comparison_json = "/opt/CFD-Benchmark/MMS/result/compare.json"  # Output JSON with comparison results

compare_with_exact_solution(input_json, output_folder, comparison_json)
