import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from diffusion_solver import DiffusionSolverCN  # Import your solver


def analytical_solution(x, t):
    """ Analytical solution of 1D diffusion equation (Manufactured Solution). """
    return np.exp(-t) * np.sin(np.pi * x)


# Define test cases with varying parameters
test_cases = []

# 1. Vary Simulation Time T (0.5 to 5.0)
for T in [0.5, 1.0, 2.0, 5.0]:
    nt = int(T / 0.001)  # Keep dt fixed, so adjust nt
    test_cases.append({"nx": 100, "nt": nt, "dx": 0.02, "dt": 0.001, "nu": 0.1, "T": T})

# 2. Vary Grid Resolution
for nx, nt, dx, dt in [(50, 100, 0.04, 0.002), (100, 200, 0.02, 0.001),
                       (200, 400, 0.01, 0.0005), (300, 600, 0.0067, 0.00033)]:
    test_cases.append({"nx": nx, "nt": nt, "dx": dx, "dt": dt, "nu": 0.1, "T": nt * dt})

# 3. Vary Diffusivity (ν)
for nu in [0.01, 0.1, 1.0]:
    test_cases.append({"nx": 100, "nt": 200, "dx": 0.02, "dt": 0.001, "nu": nu, "T": 0.2})

# Store results
results = []
text_results = []  # Store text output for saving

for case in test_cases:
    nx, nt, dx, dt, nu, T = case["nx"], case["nt"], case["dx"], case["dt"], case["nu"], case["T"]
    test_description = f"Running test case: T={T}, nx={nx}, nt={nt}, dx={dx}, dt={dt}, nu={nu}"
    print(test_description)
    text_results.append(test_description)

    # Initialize solver
    solver = DiffusionSolverCN(nx, nt, dx, dt, nu)

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

    test_result = f"L2 Norm Error: {l2_norm_error:.6f}, Max Absolute Error: {max_error:.6f}, Execution Time: {execution_time:.3f} sec\n"
    print(test_result)
    text_results.append(test_result)

    # Store results in a list
    results.append({
        "T": T,
        "nx": nx,
        "nt": nt,
        "dx": dx,
        "dt": dt,
        "nu": nu,
        "L2 Error": l2_norm_error,
        "Max Error": max_error,
        "Execution Time (s)": execution_time
    })

# Save results as a text file
with open("diffusion_performance_results.txt", "w") as f:
    f.writelines("\n".join(text_results))

# Save structured results as a CSV file
df_results = pd.DataFrame(results)
df_results.to_csv("diffusion_performance_results.csv", index=False)

print("\nAll results have been saved to 'diffusion_performance_results.txt' and 'diffusion_performance_results.csv'")

# ========== Plot Performance ==========
plt.figure(figsize=(10, 5))

# Plot L2 Error vs Grid Resolution
plt.subplot(1, 2, 1)
plt.plot([case["dx"] for case in test_cases], [res["L2 Error"] for res in results], marker="o", linestyle="--",
         label="L2 Error")
plt.xlabel("Grid Resolution (dx)")
plt.ylabel("L2 Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("L2 Error vs Grid Resolution")
plt.grid(True)

# Plot Execution Time vs Problem Size
plt.subplot(1, 2, 2)
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
