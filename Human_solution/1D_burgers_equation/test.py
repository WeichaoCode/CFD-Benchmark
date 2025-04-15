import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from bugers_solver import BurgersSolverCN  # Import your solver


def analytical_solution(x, t):
    """ Analytical solution of Burgers' equation (Manufactured Solution). """
    return np.exp(-t) * np.sin(np.pi * x)


# Define test cases with different x-domain, t-domain, and grid resolutions
test_cases = [
    {"nx": 50, "nt": 100, "dx": 0.04, "dt": 0.002, "nu": 0.1},
    {"nx": 100, "nt": 200, "dx": 0.02, "dt": 0.001, "nu": 0.1},
    {"nx": 200, "nt": 400, "dx": 0.01, "dt": 0.0005, "nu": 0.1},
    {"nx": 300, "nt": 600, "dx": 0.0067, "dt": 0.00033, "nu": 0.1},
]

# Store results
results = []

for case in test_cases:
    nx, nt, dx, dt, nu = case["nx"], case["nt"], case["dx"], case["dt"], case["nu"]
    print(f"Running test case: nx={nx}, nt={nt}, dx={dx}, dt={dt}, nu={nu}")

    # Initialize solver
    solver = BurgersSolverCN(nx, nt, dx, dt, nu)

    # Define spatial and temporal grid
    x = np.linspace(0, (nx - 1) * dx, nx)
    t = np.linspace(0, (nt - 1) * dt, nt)

    # Initial condition
    initial_condition = analytical_solution(x, 0)
    solver.initialize(initial_condition)

    # Time the execution
    start_time = time.time()
    solver.solve()
    execution_time = time.time() - start_time

    # Compute the analytical solution
    analytical_sol = np.zeros((nx, nt))
    for n in range(nt):
        analytical_sol[:, n] = analytical_solution(x, t[n])

    # Compute errors
    numerical_solution = solver.get_solution()
    error = np.abs(numerical_solution - analytical_sol)
    l2_norm_error = np.sqrt(np.sum(error ** 2) / np.sum(analytical_sol ** 2))  # L2 norm error
    max_error = np.max(error)  # Maximum absolute error

    print(
        f"L2 Norm Error: {l2_norm_error:.6f}, Max Absolute Error: {max_error:.6f}, Execution Time: {execution_time:.3f} sec\n")

    # Store results
    results.append({
        "nx": nx,
        "nt": nt,
        "dx": dx,
        "dt": dt,
        "nu": nu,
        "L2 Error": l2_norm_error,
        "Max Error": max_error,
        "Execution Time (s)": execution_time
    })

# Convert results to DataFrame and save as CSV
df_results = pd.DataFrame(results)
df_results.to_csv("performance_results.csv", index=False)

# Plot error vs grid resolution
plt.figure(figsize=(10, 5))
plt.plot([case["dx"] for case in test_cases], [res["L2 Error"] for res in results], marker="o", linestyle="--",
         label="L2 Error")
plt.plot([case["dx"] for case in test_cases], [res["Max Error"] for res in results], marker="s", linestyle="--",
         label="Max Error")
plt.xlabel("Grid Resolution (dx)")
plt.ylabel("Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Error vs Grid Resolution")
plt.grid(True)
plt.show()

# Plot execution time vs problem size
plt.figure(figsize=(10, 5))
plt.plot([case["nx"] * case["nt"] for case in test_cases], [res["Execution Time (s)"] for res in results], marker="o",
         linestyle="--", label="Execution Time")
plt.xlabel("Problem Size (nx * nt)")
plt.ylabel("Execution Time (s)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Execution Time vs Problem Size")
plt.grid(True)
plt.show()
