import numpy as np
import matplotlib.pyplot as plt

# define grid parameters
nx = 101  # number of spatial grid points
nt = 101  # number of time steps
dx = 2.0 / (nx - 1)  # spatial grid size
dt = 2.0 / (nt - 1)  # time step size
c = 1.0  # wave speed

# initialize solution arrays
u = np.zeros((nt, nx))
x = np.zeros(nx)

# set initial conditions
for i in range(nx):
    x[i] = i * dx
    u[0, i] = np.sin(np.pi * x[i])

# apply boundary conditions
u[:, 0] = 0
u[:, -1] = 0

# solve PDE
for n in range(nt - 2):
    for i in range(2, nx):
        u[n+1, i] = u[n, i] - 0.5*c*dt/dx*(3*u[n, i] - 4*u[n, i-1] + u[n, i-2]) + \
        2*(dt**2)/(dx**2)*(u[n, i-2] - 2*u[n, i-1] + u[n, i]) - \
        dt*np.pi*c*np.exp(-n*dt)*np.cos(np.pi*x[i]) + dt*np.exp(-n*dt)*np.sin(np.pi*x[i])

# plot solution
plt.figure(figsize=(10, 6))
plt.plot(x, u[0, :], label="t = 0")
plt.plot(x, u[nt//4, :], label="t = T/4")
plt.plot(x, u[nt//2, :], label="t = T/2")
plt.plot(x, u[-1, :], label="t = T")
plt.title("1D Linear Convection - Beam-Warming Method")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
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
# T = 2.0
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
# plt.savefig("/opt/CFD-Benchmark/MMS/generated_code/gpt-4/2.png")
# plt.show()