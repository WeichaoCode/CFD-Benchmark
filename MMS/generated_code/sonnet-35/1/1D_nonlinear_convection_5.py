import numpy as np
import matplotlib.pyplot as plt

def source_term(x, t):
    """Source term in the equation"""
    return np.exp(-t) * np.sin(np.pi*x) - np.pi * np.exp(-2*t) * np.sin(np.pi*x) * np.cos(np.pi*x)

# Grid parameters
nx = 100  # number of spatial points
nt = 1000  # number of time steps
dx = 2.0/(nx-1)  # spatial step size
dt = 2.0/nt  # time step size
x = np.linspace(0, 2, nx)  # spatial domain
t = 0  # initial time

# Stability condition (CFL)
c = dt/dx
if c > 1:
    print("Warning: CFL condition not satisfied!")

# Initial condition
u = np.sin(np.pi*x)

# Arrays to store solutions at key time steps
u_t0 = u.copy()
u_t1 = np.zeros_like(u)
u_t2 = np.zeros_like(u)
u_t3 = np.zeros_like(u)

# Time stepping
for n in range(nt):
    # Store solutions at key time steps
    if abs(n*dt - 0.5) < dt/2:  # t = T/4
        u_t1 = u.copy()
    elif abs(n*dt - 1.0) < dt/2:  # t = T/2
        u_t2 = u.copy()
    elif abs(n*dt - 2.0) < dt/2:  # t = T
        u_t3 = u.copy()

    # Temporary array for new time step
    un = u.copy()

    # Lax-Friedrichs scheme
    for i in range(1, nx-1):
        u[i] = 0.5*(un[i+1] + un[i-1]) - \
               0.5*c*(un[i+1]**2 - un[i-1]**2)/2 + \
               dt*source_term(x[i], n*dt)

    # Boundary conditions
    u[0] = 0
    u[-1] = 0

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, u_t0, 'b-', label='t = 0')
plt.plot(x, u_t1, 'r--', label='t = T/4')
plt.plot(x, u_t2, 'g-.', label='t = T/2')
plt.plot(x, u_t3, 'k:', label='t = T')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Nonlinear Convection - Lax-Friedrichs Method')
plt.legend()
plt.grid(True)
plt.show()

########################################################################################
# The following codes are added by user
########################################################################################
# Function to compute the exact solution
def exact_solution(x, t):
    return np.exp(-t) * np.sin(np.pi * x)


# Function to compute L2 norm and MSE
def compute_errors(numerical, exact):
    l2_norm = np.linalg.norm(numerical - exact)
    mse = np.mean((numerical - exact) ** 2)
    return l2_norm, mse


# Plotting
T = 2.0
plt.figure(figsize=(8, 6))
time_steps = [0, 1, 2, 3]
u = [u_t0, u_t1, u_t2, u_t3]
time_labels = [0, T / 4, T / 2, T]
for idx, t_step in enumerate(time_steps):
    t = time_labels[idx]

    # Numerical and Exact solutions
    u_num = u[t_step]
    u_exact = exact_solution(x, t)

    # Compute L2 and MSE errors
    l2_error, mse_error = compute_errors(u_num, u_exact)

    # Plot numerical and exact solutions
    plt.plot(x, u_num, label=f'Numerical (t = {t})')
    plt.plot(x, u_exact, '--', label=f'Exact (t = {t})')

    # Display error metrics
    print(f"Time t = {t}: L2 Error = {l2_error:.4e}, MSE = {mse_error:.4e}")

plt.title('1D Linear Convection - Numerical vs Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.savefig("/opt/CFD-Benchmark/MMS/generated_code/sonnet-35/5.png")
plt.show()