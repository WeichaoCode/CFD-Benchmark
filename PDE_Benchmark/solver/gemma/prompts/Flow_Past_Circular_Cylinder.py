import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
r_min = 0.5
r_max = 10
theta_min = 0
theta_max = 2*np.pi
t_max = 10
dt = 0.01
nu = 0.005
v_inf = 1

# Spatial discretization
Nr = 100
Ntheta = 100
dr = (r_max - r_min) / (Nr - 1)
dtheta = (theta_max - theta_min) / (Ntheta - 1)

# Time discretization
Nt = int(t_max / dt)

# Initialize arrays
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
psi = np.zeros((Nr, Ntheta))
omega = np.zeros((Nr, Ntheta))

# Boundary conditions
psi[:, 0] = 20
psi[:, -1] = v_inf * r + 20
omega[:, 0] = 2 * (psi[:, 0] - psi[:, 1]) / dr**2
omega[:, -1] = 0

# Initial conditions
psi[:, :] = 0
omega[:, :] = 0

# Velocity field initialization
u_r = np.zeros_like(psi)
u_theta = np.zeros_like(psi)

# Time loop
for n in range(Nt):
    # Calculate velocity components
    u_r = (1 / r) * np.diff(psi, axis=1)
    u_theta = -np.diff(psi, axis=0)

    # Update vorticity
    omega_new = omega + dt * (-u_r * np.diff(omega, axis=1) / dr - u_theta * np.diff(omega, axis=0) / r * (1 / r)) + dt * nu * np.diff(np.diff(omega, axis=0), axis=0) + dt * nu * np.diff(np.diff(omega, axis=1), axis=1)

    # Update streamfunction
    psi_new = psi + dt * (u_r * np.diff(psi, axis=1) / dr + u_theta * np.diff(psi, axis=0) / r * (1 / r))

    # Apply boundary conditions
    psi_new[:, 0] = 20
    psi_new[:, -1] = v_inf * r + 20
    omega_new[:, 0] = 2 * (psi_new[:, 0] - psi_new[:, 1]) / dr**2
    omega_new[:, -1] = 0

    # Update arrays
    psi = psi_new
    omega = omega_new

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)