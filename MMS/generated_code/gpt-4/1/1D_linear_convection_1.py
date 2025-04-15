import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
c = 1.0
T = 2.0
L = 2.0
nx = 101
dx = L / (nx - 1)
dt = 0.005  # This satisfies the CFL condition for FOU method
nt = int(T / dt + 1)

# Create a structured mesh
x = np.linspace(0, L, nx)

# Initialize the solution array
u = np.zeros((nt, nx))

# Apply the initial conditions
u[0, :] = np.sin(np.pi * x)

# Time stepping loop
for n in range(nt - 1):
    t = n * dt
    u[n + 1, 1:] = u[n, 1:] - c * dt / dx * (u[n, 1:] - u[n, :-1]) + dt * np.pi * c * np.exp(-t) * np.cos(
        np.pi * x[1:]) - dt * np.exp(-t) * np.sin(np.pi * x[1:])

# Apply the boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# Plot the solution at key time steps
plt.figure(figsize=(6, 4))
plt.plot(x, u[0, :], label='t = 0')
plt.plot(x, u[nt // 4, :], label='t = T/4')
plt.plot(x, u[nt // 2, :], label='t = T/2')
plt.plot(x, u[-1, :], label='t = T')
plt.title('1D Linear Convection - First Order Upwind Method')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# ########################################################################################
# # The following codes are added by user
# ########################################################################################
# # Function to compute the exact solution
# def exact_solution(x, t):
#     return np.exp(-t) * np.sin(np.pi * x)
#
#
# # Function to compute L2 norm and MSE
# def compute_errors(numerical, exact):
#     l2_norm = np.linalg.norm(numerical - exact)
#     mse = np.mean((numerical - exact) ** 2)
#     return l2_norm, mse
#
#
# # Plotting
# plt.figure(figsize=(8, 6))
# time_steps = [0, nt // 4, nt // 2, -1]
# time_labels = [0, T / 4, T / 2, T]
# for idx, t_step in enumerate(time_steps):
#     t = time_labels[idx]
#
#     # Numerical and Exact solutions
#     u_num = u[t_step, :]
#     u_exact = exact_solution(x, t)
#
#     # Compute L2 and MSE errors
#     l2_error, mse_error = compute_errors(u_num, u_exact)
#
#     # Plot numerical and exact solutions
#     plt.plot(x, u_num, label=f'Numerical (t = {t})')
#     plt.plot(x, u_exact, '--', label=f'Exact (t = {t})')
#
#     # Display error metrics
#     print(f"Time t = {t}: L2 Error = {l2_error:.4e}, MSE = {mse_error:.4e}")
#
# plt.title('1D Linear Convection - Numerical vs Exact Solution')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.legend()
# plt.grid(True)
# plt.savefig("/opt/CFD-Benchmark/MMS/generated_code/gpt-4/1.png")
# plt.show()