import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_min, x_max = -5.0, 5.0
N_x = 101
c = 1.0
epsilons = [0.0, 5e-4]
T = 5.0  # Final time
CFL = 0.4

# Spatial discretization
x = np.linspace(x_min, x_max, N_x)
dx = x[1] - x[0]

# Time step based on CFL condition
dt = CFL * dx / c
N_t = int(T / dt)
dt = T / N_t  # Recalculate dt to exactly reach T

# Initial condition
u_initial = np.exp(-x**2)

# Function to compute the right-hand side of the PDE
def compute_rhs(u, c, epsilon, dx):
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

# Prepare plot
plt.figure(figsize=(10, 6))
plt.plot(x, u_initial, label='Initial Condition', linestyle='--')

for epsilon in epsilons:
    # Initialize solution
    u = u_initial.copy()
    
    # Time integration using RK4
    for _ in range(N_t):
        k1 = compute_rhs(u, c, epsilon, dx)
        k2 = compute_rhs(u + 0.5 * dt * k1, c, epsilon, dx)
        k3 = compute_rhs(u + 0.5 * dt * k2, c, epsilon, dx)
        k4 = compute_rhs(u + dt * k3, c, epsilon, dx)
        u += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Save the final solution
    if epsilon == 0.0:
        np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_1D_Linear_Convection_rk.npy', u)
        label = 'Undamped (ε=0)'
    else:
        np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_1D_Linear_Convection_rk.npy', u)
        label = f'Damped (ε={epsilon})'
    
    # Plot the final solution
    plt.plot(x, u, label=label)

# Final plot settings
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title('Wave Propagation: Damped vs Undamped Cases')
plt.legend()
plt.grid(True)
plt.show()