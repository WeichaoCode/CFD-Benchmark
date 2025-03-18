import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2.0
Nx = 81
CFL = 1.0
T = 0.25
gamma = 1.4
x0 = 0.5 * L

# Discretize the domain
dx = L / (Nx - 1)
x = np.linspace(-L/2, L/2, Nx)
dt = CFL * dx / np.max(np.sqrt(gamma))

# Initialize variables
rho = np.where(x < x0, 1.0, 0.125)
u = np.zeros_like(x)
p = np.where(x < x0, 1.0, 0.1)
U = np.array([rho, rho*u, rho*(0.5*u**2 + p/(gamma-1))])

# Define helper functions
def compute_F(U):
    rho, rhou, rhoE = U
    u = rhou / rho
    p = (gamma - 1) * (rhoE - 0.5 * rho * u**2)
    return np.array([rhou, rhou*u + p, u*(rhoE + p)])

def update_primitive_variables(U):
    rho, rhou, rhoE = U
    u = rhou / rho
    p = (gamma - 1) * (rhoE - 0.5 * rho * u**2)
    return rho, u, p

# Time integration
for t in np.arange(0, T, dt):
    F = compute_F(U)
    U_star = U[:, :-1] - dt/dx * (F[:, 1:] - F[:, :-1])
    F_star = compute_F(U_star)
    U[:, 1:] = 0.5 * (U[:, 1:] + U_star - dt/dx * (F_star - F[:, 1:]))
    rho, u, p = update_primitive_variables(U)

# Visualization
plt.figure(figsize=(12, 9))
plt.subplot(311); plt.plot(x, rho); plt.ylabel('Density')
plt.subplot(312); plt.plot(x, u); plt.ylabel('Velocity')
plt.subplot(313); plt.plot(x, p); plt.ylabel('Pressure')
plt.xlabel('x')
plt.tight_layout()
plt.show()

# Save the results
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/U_1D_Euler_Shock_Tube.npy', U)