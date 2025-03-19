import numpy as np
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
import logging
from datetime import datetime

flag = None
logging.basicConfig(
    filename="/opt/CFD-Benchmark/PDE_Benchmark_6/report/compare.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Get current time with microseconds
    timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
    plt.savefig(f"/opt/CFD-Benchmark/PDE_Benchmark_6/images/{timestamp}")
    plt.show()


def plot_results_2d(output1, output2, mse):
    """
    Plots two 2D arrays (GT and PRED) and their absolute error.

    Parameters:
    - output1: Ground Truth 2D array
    - output2: Predicted 2D array
    - mse: Mean Squared Error
    """

    # Ensure output1 and output2 are 2D
    if output1.ndim != 2 or output2.ndim != 2:
        raise ValueError("Both output1 and output2 must be 2D arrays.")

    error = np.abs(output1 - output2)  # Compute Absolute Error

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Ground Truth (GT)
    im1 = axes[0].imshow(output1, cmap='jet', origin='lower')
    axes[0].set_title("GT Output")
    plt.colorbar(im1, ax=axes[0])

    # Plot Predicted (PRED)
    im2 = axes[1].imshow(output2, cmap='jet', origin='lower')
    axes[1].set_title("PRED Output")
    plt.colorbar(im2, ax=axes[1])

    # Plot Absolute Error
    im3 = axes[2].imshow(error, cmap='hot', origin='lower')
    axes[2].set_title(f"Absolute Error (MSE: {mse:.5f})")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    # Get current time with microseconds
    timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
    plt.savefig(f"/opt/CFD-Benchmark/PDE_Benchmark_6/images/{timestamp}")
    plt.show()


def main_process(py_path, npy_path):
    flag = os.path.splitext(py_path)[0]  # Remove .py from script_name
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
    if flag == "2D_Steady_Heat_Equation":
        output2 = output2.T
    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(output1.flatten(), output2.flatten())
    print(f"Mean Squared Error: {mse:.5f}")
    logging.info(f"current problem is {flag}, Mean Squared Error: {mse:.5f}")

    # Example usage
    try:
        # Attempt to plot as 2D data
        plot_results_2d(output1, output2, mse)
    except Exception as e:
        print(f"⚠️ 2D plotting failed: {e}\nSwitching to 1D plotting...")

        try:
            # Attempt to plot as 1D data
            plot_results_1d(output1, output2, mse)
        except Exception as e1:
            print(f"❌ 1D plotting also failed: {e1}")


main_process('1D_Diffusion.py', 'u_1D_Diffusion.npy')

main_process('1D_Euler_Shock_Tube.py', 'U_1D_Euler_Shock_Tube.npy')

main_process('1D_Linear_Convection_adams.py', 'u_1D_Linear_Convection_adams.npy')

main_process('1D_Linear_Convection_explicit_euler.py', 'u_1D_Linear_Convection_explicit_euler.npy')

main_process('1D_Linear_Convection_pred_corr.py', 'u_1D_Linear_Convection_pred_corr.npy')

main_process('1D_Linear_Convection_rk.py', 'u_1D_Linear_Convection_rk.npy')

main_process('1D_Nonlinear_Convection_Lax.py', 'u_1D_Nonlinear_Convection_Lax.npy')

main_process('1D_Nonlinear_Convection_LW.py', 'u_1D_Nonlinear_Convection_LW.npy')

main_process('1D_Nonlinear_Convection_Mk.py', 'u_1D_Nonlinear_Convection_Mk.npy')

main_process('2D_Burgers_Equation.py', 'u_2D_Burgers_Equation.npy')
main_process('2D_Burgers_Equation.py', 'v_2D_Burgers_Equation.npy')

main_process('2D_Convection.py', 'u_2D_Convection.npy')
main_process('2D_Convection.py', 'v_2D_Convection.npy')

main_process('2D_Diffusion.py', 'u_2D_Diffusion.npy')

main_process('2D_Diffusion_FVM.py', 'w_2D_Diffusion_FVM_SA.npy')

main_process('2D_Inviscid_Burgers_FOU.py', 'u_2D_Inviscid_Burgers_FOU.npy')
main_process('2D_Inviscid_Burgers_FOU.py', 'v_2D_Inviscid_Burgers_FOU.npy')

main_process('2D_Inviscid_Burgers_MK.py', 'u_2D_Inviscid_Burgers_MK.npy')
main_process('2D_Inviscid_Burgers_MK.py', 'v_2D_Inviscid_Burgers_MK.npy')

main_process('2D_Laplace_Equation.py', 'p_2D_Laplace_Equation.npy')

main_process('2D_Linear_Convection.py', 'u_2D_Linear_Convection.npy')

main_process('2D_Possion_Equation.py', 'p_2D_Possion.npy')

main_process('2D_Steady_Heat_Equation.py', 'T_2D_Steady_Heat_Equation.npy')

main_process('2D_Unsteady_Heat_Equation_SE.py', 'T_2D_Unsteady_Heat_Equation_SE.npy')

main_process('2D_Unsteady_Heat_Equation_ADI.py', 'T_2D_Unsteady_Heat_Equation_ADI.npy')

main_process('2D_Navier_Stokes_Cavity.py', 'u_2D_Navier_Stokes_Cavity.npy')
main_process('2D_Navier_Stokes_Cavity.py', 'v_2D_Navier_Stokes_Cavity.npy')
main_process('2D_Navier_Stokes_Cavity.py', 'p_2D_Navier_Stokes_Cavity.npy')

main_process('2D_Navier_Stokes_Channel.py', 'u_2D_Navier_Stokes_Channel.npy')
main_process('2D_Navier_Stokes_Channel.py', 'v_2D_Navier_Stokes_Channel.npy')
main_process('2D_Navier_Stokes_Channel.py', 'p_2D_Navier_Stokes_Channel.npy')
