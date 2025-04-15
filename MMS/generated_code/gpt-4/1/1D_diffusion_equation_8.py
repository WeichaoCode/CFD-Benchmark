import numpy as np
import matplotlib.pyplot as plt

# Define constants
nu = 0.3
pi = np.pi
x_start, x_end = 0, 2
t_start, t_end = 0, 2
N = 100  # number of mesh points
dx = (x_end - x_start) / (N - 1)  # space step
dt = dx  # time step, assuming dx=dt
alpha = nu * dt / dx**2  # diffusion number

# Check stability conditions
if alpha < 0.5:
    print('Solution is stable')
else:
    print('Solution may be unstable')

# Initialize solution: at t=0, u=sin(pi*x)
x = np.linspace(x_start, x_end, N)
u = np.sin(pi * x)
u_list = []
# Time stepping 
for t in np.arange(t_start + dt, t_end + dt, dt):
    un = u.copy()
    u[1:-1] = (un[1:-1] - dt * pi**2 * nu * np.exp(-t) * np.sin(pi * x[1:-1]) + dt * np.exp(-t) * np.sin(pi * x[1:-1]) +
               alpha * (un[:-2] - 2*un[1:-1] + un[2:]))
    u[0] = u[-1] = 0  # boundary conditions

    # Plot solution at certain time steps
    if t in [t_start, t_end/4, t_end/2, t_end]:
        u_list.append(u)
        plt.plot(x, u, label=f't={t:.2f}')

# Configure and show plot
plt.title('1D Diffusion Equation via Lax-Friedrichs Method')
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
# nt = N
# T = 2.0
# plt.figure(figsize=(8, 6))
# time_steps = [0, nt // 4, nt // 2, -1]
# time_labels = [0, T / 4, T / 2, T]
# for idx, t_step in enumerate(time_steps):
#     t = time_labels[idx]
#
#     # Numerical and Exact solutions
#     u_num = u_list[idx]
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