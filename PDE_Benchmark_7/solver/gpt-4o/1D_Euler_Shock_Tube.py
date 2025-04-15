import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2.0          # Tube length
N_x = 81         # Number of spatial points
gamma = 1.4      # Ratio of specific heats
CFL = 1.0        # CFL number
T = 0.25         # Total simulation time

# Discretize the spatial domain
x = np.linspace(-1.0, 1.0, N_x)
dx = x[1] - x[0]

# Initial conditions
rho = np.ones(N_x)
u = np.zeros(N_x)
p = np.ones(N_x)
E = np.ones(N_x)

# Initialize left and right states
rho[x >= 0] = 0.125
p[x >= 0] = 0.1

# Convert initial conditions to conservative variables
E = p / ((gamma - 1) * rho) + 0.5 * u**2
U = np.zeros((3, N_x))
U[0] = rho
U[1] = rho * u
U[2] = rho * E

# Time integration
t = 0.0
dt = CFL * dx / np.max(np.abs(u) + np.sqrt(gamma * p / rho))

while t < T:
    # Calculate flux F
    F = np.zeros((3, N_x))
    F[0] = U[1]
    F[1] = U[1]**2 / U[0] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0])
    F[2] = (U[1] / U[0]) * (U[2] + (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0]))
    
    # Predictor step
    U_pred = np.copy(U)
    U_pred[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])
    
    # Calculate flux F* for predicted values
    F_pred = np.zeros((3, N_x))
    F_pred[0] = U_pred[1]
    F_pred[1] = U_pred[1]**2 / U_pred[0] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0])
    F_pred[2] = (U_pred[1] / U_pred[0]) * (U_pred[2] + (gamma - 1) * (U_pred[2] - 0.5 * U_pred[1]**2 / U_pred[0]))
    
    # Corrector step
    U[:, 1:] = 0.5 * (U[:, 1:] + U_pred[:, 1:] - dt / dx * (F_pred[:, 1:] - F_pred[:, :-1]))
    
    # Apply reflective boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    
    # Update time
    t += dt

# Convert conservative back to primitive variables for visualization
rho = U[0]
u = U[1] / U[0]
E = U[2] / U[0]
p = (gamma - 1) * (U[2] - 0.5 * U[1]**2 / U[0])

# Save results in .npy format
np.save('/PDE_Benchmark_7/results/prediction/rho_1D_Euler_Shock_Tube.npy', rho)
np.save('/PDE_Benchmark_7/results/prediction/u_1D_Euler_Shock_Tube.npy', u)
np.save('/PDE_Benchmark_7/results/prediction/E_1D_Euler_Shock_Tube.npy', E)
# np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/F_1D_Euler_Shock_Tube.npy', F)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(x, rho)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density Profile')

plt.subplot(1, 3, 2)
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('Velocity')
plt.title('Velocity Profile')

plt.subplot(1, 3, 3)
plt.plot(x, p)
plt.xlabel('x')
plt.ylabel('Pressure')
plt.title('Pressure Profile')

plt.tight_layout()
plt.show()