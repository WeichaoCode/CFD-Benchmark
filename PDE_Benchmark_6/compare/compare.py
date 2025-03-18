import numpy as np
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

flag = None

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
SOLUTION_DIR = os.path.join(ROOT_DIR, 'solution')
SOLVER_DIR = os.path.join(ROOT_DIR, 'solver')
RESULT_DIR = os.path.join(ROOT_DIR, 'results')
TRUE_DIR = os.path.join(RESULT_DIR, 'ground_truth')
PRED_DIR = os.path.join(RESULT_DIR, 'prediction')


# Plot the two outputs and their error for 1D data
def plot_results_1d(output1, output2, mse):
    x = np.arange(len(output1))  # Create x-axis values

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot Solver 1 Output
    axes[0].plot(x, output1, label="GT", color='blue')
    axes[0].set_title("GT Output")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Solver 2 Output
    axes[1].plot(x, output2, label="PRED", color='green')
    axes[1].set_title("PRED Output")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True)

    # Plot Absolute Error
    error = np.abs(output1 - output2)
    axes[2].plot(x, error, label=f"Absolute Error: {mse:.5f}", color='red')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Error")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def main_process(py_path, npy_path):
    # Paths to the two Python scripts that solve the PDE
    solver1 = os.path.join(SOLUTION_DIR, py_path)
    solver2 = os.path.join(SOLVER_DIR, py_path)

    # Run the two solvers and save their outputs
    subprocess.run(["python", solver1])
    subprocess.run(["python", solver2])

    # Load the output data from the two solvers (assuming they save results as .npy)
    result1 = os.path.join(TRUE_DIR, npy_path)
    result2 = os.path.join(PRED_DIR, npy_path)
    output1 = np.load(result1)
    output2 = np.load(result2)

    if flag == "1D_Euler_Shock_Tube":
        rho1, u1, E1 = output1[:, 0], output1[:, 1], output1[:, 2]
        rho2, u2, E2 = output2[0, :], output2[1, :], output2[2, :]
        # Compute Mean Squared Error (MSE)
        mse_rho = mean_squared_error(rho1.flatten(), rho2.flatten())
        mse_u = mean_squared_error(u1.flatten(), u2.flatten())
        mse_E = mean_squared_error(E1.flatten(), E2.flatten())
        print(f"Mean Squared Error: rho: {mse_rho:.5f}; u: {mse_u:.5f}; E: {mse_E:.5f}")
        plot_results_1d(rho1, rho2, mse_rho)
        plot_results_1d(u1, u2, mse_u)
        plot_results_1d(E1, E2, mse_E)
        return 0
    if flag == "1D_Linear_Convection_adams":
        output2 = output2[0, -1, :]
    if flag == "1D_Linear_Convection_pred_corr":
        output2 = output2[-1, :, 0]
    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(output1.flatten(), output2.flatten())
    print(f"Mean Squared Error: {mse:.5f}")

    # Example usage
    plot_results_1d(output1, output2, mse)


# flag = '1D_Burgers_Equation'
# if flag == '1D_Burgers_Equation':
#     main_process('1D_Burgers_Equation.py', 'u_1D_Burgers_Equation.npy')

# main_process('1D_Diffusion.py', 'u_1D_Diffusion.npy')

# main_process('1D_Euler_Shock_Tube.py', 'U_1D_Euler_Shock_Tube.npy')

# main_process('1D_Linear_Convection_adams.py', 'u_1D_Linear_Convection_adams.npy')

# main_process('1D_Linear_Convection_explicit_euler.py', 'u_1D_Linear_Convection_explicit_euler.npy')

# main_process('1D_Linear_Convection_pred_corr.py', 'u_1D_Linear_Convection_pred_corr.npy')

# main_process('1D_Linear_Convection_rk.py', 'u_1D_Linear_Convection_rk.npy')

# main_process('1D_Nonlinear_Convection_Lax.py', 'u_1D_Nonlinear_Convection_Lax.npy')

# main_process('1D_Nonlinear_Convection_LW.py', 'u_1D_Nonlinear_Convection_LW.npy')

# main_process('1D_Nonlinear_Convection_Mk.py', 'u_1D_Nonlinear_Convection_Mk.npy')
