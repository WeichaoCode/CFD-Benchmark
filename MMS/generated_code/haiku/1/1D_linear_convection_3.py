import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    return np.sin(np.pi * x)

def exact_solution(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

def beam_warming_method(c, dx, dt, x, T):
    # Grid setup
    nx = len(x)
    nt = int(T / dt) + 1
    
    # Initialize solution array
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = initial_condition(x)
    
    # Beam-Warming coefficients
    alpha = c * dt / dx
    
    # Time-stepping
    for n in range(nt - 1):
        t = n * dt
        
        # Apply boundary conditions
        u[n, 0] = 0
        u[n, -1] = 0
        
        # Beam-Warming scheme
        for j in range(2, nx - 1):
            source_term = -np.pi * c * np.exp(-t) * np.cos(np.pi * x[j]) + np.exp(-t) * np.sin(np.pi * x[j])
            
            u[n+1, j] = (
                u[n, j] - 
                alpha * (3/2 * u[n, j] - 2 * u[n, j-1] + 1/2 * u[n, j-2]) +
                dt * source_term
            )
    
    return u

# Simulation parameters
c = 1.0  # wave speed
x_start, x_end = 0, 2
dx = 0.02
x = np.arange(x_start, x_end + dx, dx)
T = 2
dt = 0.01

# Solve PDE
solution = beam_warming_method(c, dx, dt, x, T)

# Plot solution at key time steps
plt.figure(figsize=(10, 6))
time_steps = [0, int(T/4/dt), int(T/2/dt), int(T/dt)]
labels = [f't = {T*i/len(time_steps):.2f}' for i in range(len(time_steps))]

for i, step in enumerate(time_steps):
    plt.plot(x, solution[step, :], label=labels[i])

plt.title('1D Linear Convection - Beam-Warming Method')
plt.xlabel('x')
plt.ylabel('u')
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

nt = int(T / dt) + 1
# Plotting
T = 2.0
plt.figure(figsize=(8, 6))
time_steps = [0, nt // 4, nt // 2, -1]
time_labels = [0, T / 4, T / 2, T]
for idx, t_step in enumerate(time_steps):
    t = time_labels[idx]

    # Numerical and Exact solutions
    u_num = solution[t_step, :]
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
plt.savefig("/opt/CFD-Benchmark/MMS/generated_code/haiku/3.png")
plt.show()