import math

# Constants
nu = 0.3
pi = 3.1416
T = 2.0
x_size = 2.0
n_timesteps = 100
n_xsteps = 100

# Mesh
dx = x_size / n_xsteps
dt = T / n_timesteps
x = [i * dx for i in range(n_xsteps+1)]
t = [i * dt for i in range(n_timesteps+1)]

# Initial and boundary conditions
u = [[0.0]*(n_xsteps+1) for _ in range(n_timesteps+1)]
for i in range(n_xsteps+1):
    u[0][i] = math.sin(pi*x[i])

# Time stepping
for n in range(n_timesteps):
    for i in range(1, n_xsteps):
        term1 = nu * (u[n][i+1] - 2*u[n][i] + u[n][i-1]) / dx**2
        term2 = pi**2 * nu * math.exp(-t[n]) * math.sin(pi * x[i])
        term3 = math.exp(-t[n]) * math.sin(pi * x[i])
        u[n+1][i] = u[n][i] + dt * (term1 - term2 + term3)

# Plotting
import matplotlib.pyplot as plt

plt.plot(x, u[0], label='t = 0')
plt.plot(x, u[n_timesteps//4], label='t = T/4')
plt.plot(x, u[n_timesteps//2], label='t = T/2')
plt.plot(x, u[n_timesteps], label='t = T')
plt.legend()
plt.title('1D Diffusion Equation + First Order Upwind')
plt.show()

# ########################################################################################
# # The following codes are added by user
# ########################################################################################
# # Function to compute the exact solution
# import numpy as np
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
# nt = n_timesteps
# time_steps = [0, nt // 4, nt // 2, -1]
# time_labels = [0, T / 4, T / 2, T]
# for idx, t_step in enumerate(time_steps):
#     t = time_labels[idx]
#
#     # Numerical and Exact solutions
#     u_num = u[t_step]
#     u_exact = exact_solution(np.array(x), t)
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
# plt.savefig("/opt/CFD-Benchmark/MMS/generated_code/gpt-4/7.png")
# plt.show()